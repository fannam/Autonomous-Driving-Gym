from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ..core.runtime_config import get_config_path, load_runtime_config
from ..core.settings import EVALUATION_CONFIG, TRAIN_CONFIG, load_stage_config
from ..core.types import GenerationSummary, TrajectoryBatch
from ..environment.config import init_env
from ..environment.reward import EvolutionaryRewardWrapper
from ..network.actor_critic import build_actor_critic
from .evolution import (
    aggregate_policy_metrics,
    clone_state_dict,
    evolve_population,
    initialize_population,
    rank_policy_aggregates,
)
from .rollout import run_episode, run_rollout_episode


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _get_training_device(device: str | None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but CUDA is not available.")
    return device_obj


def _configure_torch_runtime(device: torch.device) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if device.type != "cuda":
        return

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is not None:
        matmul_backend = getattr(cuda_backend, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = True

    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        cudnn_backend.allow_tf32 = True


def _compute_gae(
    *,
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if index == len(rewards) - 1:
            next_non_terminal = 1.0 - float(dones[index])
            next_values = float(next_value)
        else:
            next_non_terminal = 1.0 - float(dones[index])
            next_values = float(values[index + 1])
        delta = (
            float(rewards[index])
            + float(gamma) * next_values * next_non_terminal
            - float(values[index])
        )
        last_advantage = (
            delta
            + float(gamma) * float(gae_lambda) * next_non_terminal * last_advantage
        )
        advantages[index] = last_advantage
    returns = advantages + values.astype(np.float32, copy=False)
    return advantages, returns


class PPOEvolutionaryTrainer:
    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        device: str | None = "auto",
        verbose: bool = True,
    ) -> None:
        self.config_path = get_config_path(config_path)
        self.raw_config = load_runtime_config(self.config_path)
        self.config = load_stage_config("train", raw_config=self.raw_config)
        self.evaluation_config = load_stage_config("evaluation", raw_config=self.raw_config)
        self.device = _get_training_device(device)
        _configure_torch_runtime(self.device)
        self.verbose = bool(verbose)
        self.network = build_actor_critic(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(self.config.ppo.learning_rate),
        )

        self.metrics_history: list[dict[str, Any]] = []
        self.best_fitness = float("-inf")
        self.best_distance = float("-inf")
        self.best_policy_state_dict = clone_state_dict(self.network.state_dict())
        self.current_generation = 0

    def _resolve_artifact_path(self, path: str | Path) -> Path:
        raw_path = Path(path).expanduser()
        if raw_path.is_absolute():
            return raw_path.resolve()
        return (PACKAGE_ROOT / raw_path).resolve()

    def _reset_metrics_log(self) -> Path:
        metrics_path = self._resolve_artifact_path(self.config.logging.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")
        return metrics_path

    def _append_metrics_log(self, summary: GenerationSummary) -> None:
        metrics_path = self._resolve_artifact_path(self.config.logging.metrics_path)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary.to_dict(), ensure_ascii=True) + "\n")

    def _prepare_training_tensors(
        self,
        trajectories: list[TrajectoryBatch],
    ) -> dict[str, torch.Tensor]:
        if not trajectories:
            raise ValueError("Cannot prepare PPO tensors without trajectories.")

        obs_chunks: list[np.ndarray] = []
        action_chunks: list[np.ndarray] = []
        behavior_log_prob_chunks: list[np.ndarray] = []
        return_chunks: list[np.ndarray] = []
        advantage_chunks: list[np.ndarray] = []

        self.network.eval()
        with torch.no_grad():
            for trajectory in trajectories:
                if trajectory.obs.size == 0:
                    continue

                obs_tensor = torch.as_tensor(
                    trajectory.obs,
                    dtype=torch.float32,
                    device=self.device,
                )
                _, values = self.network(obs_tensor, return_logits=True)
                next_obs_tensor = torch.as_tensor(
                    trajectory.last_obs[None, ...],
                    dtype=torch.float32,
                    device=self.device,
                )
                _, next_value = self.network(next_obs_tensor, return_logits=True)

                values_np = values.squeeze(-1).detach().cpu().numpy().astype(np.float32, copy=False)
                bootstrap_value = 0.0 if bool(trajectory.dones[-1]) else float(next_value.item())
                advantages, returns = _compute_gae(
                    rewards=trajectory.rewards.astype(np.float32, copy=False),
                    dones=trajectory.dones.astype(np.float32, copy=False),
                    values=values_np,
                    next_value=bootstrap_value,
                    gamma=float(self.config.ppo.gamma),
                    gae_lambda=float(self.config.ppo.gae_lambda),
                )

                obs_chunks.append(trajectory.obs.astype(np.float32, copy=False))
                action_chunks.append(trajectory.actions.astype(np.int64, copy=False))
                behavior_log_prob_chunks.append(
                    trajectory.behavior_log_probs.astype(np.float32, copy=False)
                )
                return_chunks.append(returns.astype(np.float32, copy=False))
                advantage_chunks.append(advantages.astype(np.float32, copy=False))

        obs = np.concatenate(obs_chunks, axis=0)
        actions = np.concatenate(action_chunks, axis=0)
        behavior_log_probs = np.concatenate(behavior_log_prob_chunks, axis=0)
        returns = np.concatenate(return_chunks, axis=0)
        advantages = np.concatenate(advantage_chunks, axis=0)
        advantages = (advantages - advantages.mean()) / max(1e-8, float(advantages.std()))

        return {
            "obs": torch.as_tensor(obs, dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(actions, dtype=torch.long, device=self.device),
            "behavior_log_probs": torch.as_tensor(
                behavior_log_probs,
                dtype=torch.float32,
                device=self.device,
            ),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
        }

    def _ppo_update(self, trajectories: list[TrajectoryBatch]) -> dict[str, float]:
        tensors = self._prepare_training_tensors(trajectories)
        obs = tensors["obs"]
        actions = tensors["actions"]
        behavior_log_probs = tensors["behavior_log_probs"]
        returns = tensors["returns"]
        advantages = tensors["advantages"]

        sample_count = int(obs.shape[0])
        if sample_count == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
            }

        self.network.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_items = 0

        for _ in range(int(self.config.ppo.ppo_epochs)):
            permutation = torch.randperm(sample_count, device=self.device)
            epoch_kl_sum = 0.0
            epoch_items = 0

            for start in range(0, sample_count, int(self.config.ppo.minibatch_size)):
                batch_indices = permutation[start : start + int(self.config.ppo.minibatch_size)]
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_behavior_log_probs = behavior_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                logits, values = self.network(batch_obs, return_logits=True)
                distribution = Categorical(logits=logits)
                current_log_probs = distribution.log_prob(batch_actions)
                entropy = distribution.entropy().mean()

                log_ratio = current_log_probs - batch_behavior_log_probs
                ratio = torch.exp(log_ratio)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(self.config.ppo.clip_epsilon),
                    1.0 + float(self.config.ppo.clip_epsilon),
                ) * batch_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                loss = (
                    policy_loss
                    + float(self.config.ppo.value_coef) * value_loss
                    - float(self.config.ppo.entropy_coef) * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    max_norm=float(self.config.ppo.max_grad_norm),
                )
                self.optimizer.step()

                approx_kl = torch.mean((ratio - 1.0) - log_ratio).detach()
                batch_items = int(batch_obs.shape[0])
                total_items += batch_items
                epoch_items += batch_items
                total_policy_loss += float(policy_loss.detach().item()) * batch_items
                total_value_loss += float(value_loss.detach().item()) * batch_items
                total_entropy += float(entropy.detach().item()) * batch_items
                total_kl += float(approx_kl.item()) * batch_items
                epoch_kl_sum += float(approx_kl.item()) * batch_items

            mean_epoch_kl = epoch_kl_sum / max(1, epoch_items)
            if mean_epoch_kl > float(self.config.ppo.target_kl):
                break

        self.network.eval()
        return {
            "policy_loss": total_policy_loss / max(1, total_items),
            "value_loss": total_value_loss / max(1, total_items),
            "entropy": total_entropy / max(1, total_items),
            "approx_kl": total_kl / max(1, total_items),
        }

    def _collect_generation_rollouts(
        self,
        population: list[dict[str, torch.Tensor]],
        *,
        generation_index: int,
        workers: int,
        seed_start: int,
        executor: ProcessPoolExecutor | None = None,
    ) -> list[TrajectoryBatch]:
        episodes_per_policy = int(self.config.rollout.episodes_per_policy)
        max_steps = self.config.rollout.max_steps

        tasks = []
        for policy_index in range(len(population)):
            for episode_offset in range(episodes_per_policy):
                task_seed = (
                    int(seed_start)
                    + generation_index * len(population) * episodes_per_policy
                    + policy_index * episodes_per_policy
                    + episode_offset
                )
                tasks.append((policy_index, task_seed))

        trajectories: list[TrajectoryBatch] = []
        if int(workers) <= 1:
            for policy_index, task_seed in tasks:
                trajectories.append(
                    run_rollout_episode(
                        population[policy_index],
                        config_path=self.config_path,
                        stage="train",
                        seed=task_seed,
                        policy_index=policy_index,
                        deterministic=False,
                        max_steps=max_steps,
                    )
                )
        else:
            owns_executor = executor is None
            if owns_executor:
                spawn_context = mp.get_context("spawn")
                executor = ProcessPoolExecutor(
                    max_workers=int(workers),
                    mp_context=spawn_context,
                )
            try:
                assert executor is not None
                future_map = {
                    executor.submit(
                        run_rollout_episode,
                        population[policy_index],
                        config_path=self.config_path,
                        stage="train",
                        seed=task_seed,
                        policy_index=policy_index,
                        deterministic=False,
                        max_steps=max_steps,
                    ): (policy_index, task_seed)
                    for policy_index, task_seed in tasks
                }
                for future in as_completed(future_map):
                    trajectories.append(future.result())
            finally:
                if owns_executor and executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

        trajectories.sort(key=lambda trajectory: (trajectory.policy_index, trajectory.seed))
        return trajectories

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        output_path = self._resolve_artifact_path(path or self.config.model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "ppo_state_dict": clone_state_dict(self.network.state_dict()),
            "best_policy_state_dict": clone_state_dict(self.best_policy_state_dict),
            "generation": int(self.current_generation),
            "config_snapshot": self.raw_config,
            "metrics_history": list(self.metrics_history),
            "best_fitness": float(self.best_fitness),
            "best_distance": float(self.best_distance),
        }
        torch.save(checkpoint, output_path)
        return output_path

    def load_checkpoint(self, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
        input_path = self._resolve_artifact_path(
            checkpoint_path or self.evaluation_config.model_path
        )
        return torch.load(input_path, map_location=torch.device("cpu"))

    def fit(
        self,
        *,
        generations: int,
        population_size: int,
        workers: int | None = None,
        seed_start: int = 21,
        save_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        if int(generations) <= 0:
            raise ValueError("generations must be positive.")
        if int(population_size) <= 0:
            raise ValueError("population_size must be positive.")

        worker_count = int(workers or self.config.rollout.workers)
        self.metrics_history = []
        self.best_fitness = float("-inf")
        self.best_distance = float("-inf")
        self.best_policy_state_dict = clone_state_dict(self.network.state_dict())
        self.current_generation = 0
        self._reset_metrics_log()

        population = initialize_population(
            clone_state_dict(self.network.state_dict()),
            population_size=int(population_size),
            mutation_std=float(self.config.evolution.initial_population_noise_std),
            seed=int(seed_start),
        )

        rollout_executor: ProcessPoolExecutor | None = None
        if worker_count > 1:
            spawn_context = mp.get_context("spawn")
            rollout_executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=spawn_context,
            )

        try:
            for generation_index in range(int(generations)):
                generation_number = generation_index + 1
                if self.verbose:
                    print(
                        f"[ppo-evolutionary] generation={generation_number} "
                        f"population={population_size} workers={worker_count}",
                        flush=True,
                    )

                generation_started_at = perf_counter()
                rollout_started_at = perf_counter()
                trajectories = self._collect_generation_rollouts(
                    population,
                    generation_index=generation_index,
                    workers=worker_count,
                    seed_start=int(seed_start),
                    executor=rollout_executor,
                )
                rollout_duration_s = perf_counter() - rollout_started_at

                aggregates = aggregate_policy_metrics(trajectories, population_size)
                ranked_aggregates = rank_policy_aggregates(aggregates)
                best_policy_index = int(ranked_aggregates[0].policy_index)
                best_policy_state = clone_state_dict(population[best_policy_index])

                if (
                    ranked_aggregates[0].fitness > self.best_fitness
                    or (
                        np.isclose(ranked_aggregates[0].fitness, self.best_fitness)
                        and ranked_aggregates[0].distance_travelled > self.best_distance
                    )
                ):
                    self.best_fitness = float(ranked_aggregates[0].fitness)
                    self.best_distance = float(ranked_aggregates[0].distance_travelled)
                    self.best_policy_state_dict = best_policy_state

                ppo_started_at = perf_counter()
                ppo_metrics = self._ppo_update(trajectories)
                ppo_duration_s = perf_counter() - ppo_started_at

                episode_metrics = [trajectory.episode_metrics for trajectory in trajectories]
                summary = GenerationSummary(
                    generation=generation_number,
                    best_fitness=float(ranked_aggregates[0].fitness),
                    mean_fitness=float(np.mean([aggregate.fitness for aggregate in aggregates])),
                    fitness_std=float(np.std([metric.fitness for metric in episode_metrics])),
                    collision_rate=float(np.mean([float(metric.collided) for metric in episode_metrics])),
                    success_rate=float(np.mean([float(metric.success) for metric in episode_metrics])),
                    mean_distance=float(np.mean([metric.distance_travelled for metric in episode_metrics])),
                    mean_episode_length=float(
                        np.mean([float(metric.episode_length) for metric in episode_metrics])
                    ),
                    mean_speed_mps=float(
                        np.mean([metric.mean_speed_mps for metric in episode_metrics])
                    ),
                    mean_speed_kph=float(
                        np.mean([metric.mean_speed_kph for metric in episode_metrics])
                    ),
                    mean_normalized_speed=float(
                        np.mean([metric.mean_normalized_speed for metric in episode_metrics])
                    ),
                    mean_right_lane_score=float(
                        np.mean([metric.mean_right_lane_score for metric in episode_metrics])
                    ),
                    mean_step_reward=float(
                        np.mean([metric.mean_step_reward for metric in episode_metrics])
                    ),
                    mean_raw_env_reward=float(
                        np.mean([metric.mean_raw_env_reward for metric in episode_metrics])
                    ),
                    offroad_rate=float(np.mean([metric.offroad_rate for metric in episode_metrics])),
                    policy_loss=float(ppo_metrics["policy_loss"]),
                    value_loss=float(ppo_metrics["value_loss"]),
                    entropy=float(ppo_metrics["entropy"]),
                    approx_kl=float(ppo_metrics["approx_kl"]),
                )
                self.metrics_history.append(summary.to_dict())
                self._append_metrics_log(summary)
                self.current_generation = generation_number

                checkpoint_path = self.save_checkpoint(save_path)
                generation_duration_s = perf_counter() - generation_started_at
                sample_count = int(sum(len(trajectory.actions) for trajectory in trajectories))
                if self.verbose:
                    print(
                        "[ppo-evolutionary] "
                        f"best_fitness={summary.best_fitness:.4f} "
                        f"mean_fitness={summary.mean_fitness:.4f} "
                        f"fitness_std={summary.fitness_std:.4f} "
                        f"collision_rate={summary.collision_rate:.3f} "
                        f"success_rate={summary.success_rate:.3f} "
                        f"mean_distance={summary.mean_distance:.2f} "
                        f"mean_len={summary.mean_episode_length:.2f} "
                        f"mean_speed={summary.mean_speed_kph:.2f}kmh "
                        f"mean_norm_speed={summary.mean_normalized_speed:.3f} "
                        f"lane_score={summary.mean_right_lane_score:.3f} "
                        f"step_reward={summary.mean_step_reward:.4f} "
                        f"raw_reward={summary.mean_raw_env_reward:.4f} "
                        f"offroad_rate={summary.offroad_rate:.3f} "
                        f"policy_loss={summary.policy_loss:.4f} "
                        f"value_loss={summary.value_loss:.4f} "
                        f"samples={sample_count} "
                        f"rollout_s={rollout_duration_s:.2f} "
                        f"ppo_s={ppo_duration_s:.2f} "
                        f"total_s={generation_duration_s:.2f} "
                        f"checkpoint={checkpoint_path}",
                        flush=True,
                    )

                evolution_step = evolve_population(
                    population,
                    ranked_aggregates,
                    elite_fraction=float(self.config.evolution.elite_fraction),
                    mutation_std=float(self.config.evolution.mutation_std),
                    ppo_state_dict=clone_state_dict(self.network.state_dict()),
                    seed=int(seed_start) + 1000 + generation_index * max(1, population_size),
                )
                population = evolution_step.population
        finally:
            if rollout_executor is not None:
                rollout_executor.shutdown(wait=True, cancel_futures=True)

        return list(self.metrics_history)

    def evaluate(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        policy_source: str = "best",
        episodes: int = 1,
        seed_start: int = 101,
        render_mode: str | None = None,
    ) -> dict[str, Any]:
        if int(episodes) <= 0:
            raise ValueError("episodes must be positive.")
        if policy_source not in {"best", "ppo"}:
            raise ValueError("policy_source must be one of {'best', 'ppo'}.")

        checkpoint = self.load_checkpoint(checkpoint_path)
        state_key = (
            "best_policy_state_dict"
            if policy_source == "best"
            else "ppo_state_dict"
        )
        policy_state_dict = checkpoint[state_key]

        network = build_actor_critic(self.evaluation_config)
        network.load_state_dict(policy_state_dict)
        network.to(self.device)
        network.eval()

        env = init_env(
            seed=int(seed_start),
            stage="evaluation",
            config_path=self.config_path,
            render_mode=render_mode,
        )
        wrapped_env = EvolutionaryRewardWrapper(env, self.evaluation_config.reward)

        try:
            trajectories: list[TrajectoryBatch] = []
            for episode_offset in range(int(episodes)):
                trajectories.append(
                    run_episode(
                        wrapped_env,
                        network,
                        device=self.device,
                        deterministic=True,
                        policy_index=0,
                        seed=int(seed_start) + episode_offset,
                        max_steps=self.evaluation_config.rollout.max_steps,
                        render=bool(render_mode == "human"),
                    )
                )
        finally:
            wrapped_env.close()

        episode_metrics = [trajectory.episode_metrics.to_dict() for trajectory in trajectories]
        return {
            "policy_source": policy_source,
            "episodes": episode_metrics,
            "mean_fitness": float(np.mean([metrics["fitness"] for metrics in episode_metrics])),
            "mean_distance": float(
                np.mean([metrics["distance_travelled"] for metrics in episode_metrics])
            ),
            "collision_rate": float(
                np.mean([float(metrics["collided"]) for metrics in episode_metrics])
            ),
            "success_rate": float(
                np.mean([float(metrics["success"]) for metrics in episode_metrics])
            ),
        }


__all__ = ["PPOEvolutionaryTrainer"]

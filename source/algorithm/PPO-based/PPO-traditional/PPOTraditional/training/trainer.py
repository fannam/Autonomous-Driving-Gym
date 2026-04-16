from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ..core.runtime_config import get_config_path, load_runtime_config
from ..core.settings import load_stage_config
from ..core.types import RolloutBatch, UpdateSummary
from ..environment.config import init_env
from ..environment.reward import TraditionalRewardWrapper
from ..network.actor_critic import build_actor_critic
from .collector import VectorizedRolloutCollector, run_episode


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _get_training_device(device: str | None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but CUDA is not available.")
    return device_obj


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


class PPOTraditionalTrainer:
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
        self.verbose = bool(verbose)
        self.network = build_actor_critic(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(self.config.ppo.learning_rate),
        )

        self.metrics_history: list[dict[str, Any]] = []
        self.best_fitness = float("-inf")
        self.best_distance = float("-inf")
        self.best_policy_state_dict = _clone_state_dict(self.network.state_dict())
        self.current_update = 0
        self.total_timesteps = 0

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

    def _append_metrics_log(self, summary: UpdateSummary) -> None:
        metrics_path = self._resolve_artifact_path(self.config.logging.metrics_path)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary.to_dict(), ensure_ascii=True) + "\n")

    def _flatten_rollout_batch(self, batch: RolloutBatch) -> dict[str, torch.Tensor]:
        t_steps, n_envs = batch.actions.shape
        obs = batch.obs.reshape((t_steps * n_envs, *batch.obs.shape[2:]))
        actions = batch.actions.reshape(t_steps * n_envs)
        log_probs = batch.log_probs.reshape(t_steps * n_envs)
        advantages = batch.advantages.reshape(t_steps * n_envs)
        returns = batch.returns.reshape(t_steps * n_envs)

        normalized_advantages = (advantages - advantages.mean()) / max(
            1e-8,
            float(advantages.std()),
        )
        return {
            "obs": torch.as_tensor(obs, dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(actions, dtype=torch.long, device=self.device),
            "log_probs": torch.as_tensor(log_probs, dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(
                normalized_advantages,
                dtype=torch.float32,
                device=self.device,
            ),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=self.device),
        }

    def _ppo_update(self, batch: RolloutBatch) -> dict[str, float]:
        tensors = self._flatten_rollout_batch(batch)
        obs = tensors["obs"]
        actions = tensors["actions"]
        old_log_probs = tensors["log_probs"]
        advantages = tensors["advantages"]
        returns = tensors["returns"]

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
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                logits, values = self.network(batch_obs, return_logits=True)
                distribution = Categorical(logits=logits)
                current_log_probs = distribution.log_prob(batch_actions)
                entropy = distribution.entropy().mean()

                log_ratio = current_log_probs - batch_old_log_probs
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

    def _summarize_metrics(self, batch: RolloutBatch) -> tuple[float, float, float, float, float]:
        if batch.episode_metrics:
            mean_fitness = float(np.mean([metrics.fitness for metrics in batch.episode_metrics]))
            collision_rate = float(np.mean([float(metrics.collided) for metrics in batch.episode_metrics]))
            success_rate = float(np.mean([float(metrics.success) for metrics in batch.episode_metrics]))
            mean_distance = float(np.mean([metrics.distance_travelled for metrics in batch.episode_metrics]))
            mean_episode_length = float(
                np.mean([float(metrics.episode_length) for metrics in batch.episode_metrics])
            )
            return (
                mean_fitness,
                collision_rate,
                success_rate,
                mean_distance,
                mean_episode_length,
            )

        per_env_batch_fitness = batch.rewards.sum(axis=0)
        return (
            float(np.mean(per_env_batch_fitness)),
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        output_path = self._resolve_artifact_path(path or self.config.model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "latest_policy_state_dict": _clone_state_dict(self.network.state_dict()),
            "best_policy_state_dict": _clone_state_dict(self.best_policy_state_dict),
            "update": int(self.current_update),
            "total_timesteps": int(self.total_timesteps),
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
        updates: int,
        n_envs: int | None = None,
        steps_per_env: int | None = None,
        seed_start: int = 21,
        save_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        if int(updates) <= 0:
            raise ValueError("updates must be positive.")

        resolved_n_envs = int(n_envs or self.config.rollout.n_envs)
        resolved_steps_per_env = int(steps_per_env or self.config.rollout.steps_per_env)
        self.metrics_history = []
        self.best_fitness = float("-inf")
        self.best_distance = float("-inf")
        self.best_policy_state_dict = _clone_state_dict(self.network.state_dict())
        self.current_update = 0
        self.total_timesteps = 0
        self._reset_metrics_log()

        collector = VectorizedRolloutCollector(
            config_path=self.config_path,
            stage="train",
            n_envs=resolved_n_envs,
            seed_start=int(seed_start),
        )

        try:
            for update_index in range(int(updates)):
                if self.verbose:
                    print(
                        f"[ppo-traditional] update={update_index + 1} "
                        f"n_envs={resolved_n_envs} steps_per_env={resolved_steps_per_env}",
                        flush=True,
                    )

                batch = collector.collect(
                    self.network,
                    device=self.device,
                    steps_per_env=resolved_steps_per_env,
                    deterministic=False,
                )
                ppo_metrics = self._ppo_update(batch)
                self.total_timesteps += resolved_n_envs * resolved_steps_per_env

                (
                    mean_fitness,
                    collision_rate,
                    success_rate,
                    mean_distance,
                    mean_episode_length,
                ) = self._summarize_metrics(batch)

                if (
                    mean_fitness > self.best_fitness
                    or (
                        np.isclose(mean_fitness, self.best_fitness)
                        and mean_distance > self.best_distance
                    )
                ):
                    self.best_fitness = float(mean_fitness)
                    self.best_distance = float(mean_distance)
                    self.best_policy_state_dict = _clone_state_dict(self.network.state_dict())

                summary = UpdateSummary(
                    update=update_index + 1,
                    total_timesteps=int(self.total_timesteps),
                    best_fitness=float(self.best_fitness),
                    mean_fitness=float(mean_fitness),
                    collision_rate=float(collision_rate),
                    success_rate=float(success_rate),
                    mean_distance=float(mean_distance),
                    mean_episode_length=float(mean_episode_length),
                    policy_loss=float(ppo_metrics["policy_loss"]),
                    value_loss=float(ppo_metrics["value_loss"]),
                    entropy=float(ppo_metrics["entropy"]),
                    approx_kl=float(ppo_metrics["approx_kl"]),
                )
                self.metrics_history.append(summary.to_dict())
                self._append_metrics_log(summary)
                self.current_update = update_index + 1

                checkpoint_path = self.save_checkpoint(save_path)
                if self.verbose:
                    print(
                        "[ppo-traditional] "
                        f"mean_fitness={summary.mean_fitness:.4f} "
                        f"policy_loss={summary.policy_loss:.4f} "
                        f"value_loss={summary.value_loss:.4f} "
                        f"checkpoint={checkpoint_path}",
                        flush=True,
                    )
        finally:
            collector.close()

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
        if policy_source not in {"best", "latest"}:
            raise ValueError("policy_source must be one of {'best', 'latest'}.")

        checkpoint = self.load_checkpoint(checkpoint_path)
        state_key = (
            "best_policy_state_dict"
            if policy_source == "best"
            else "latest_policy_state_dict"
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
        wrapped_env = TraditionalRewardWrapper(env, self.evaluation_config.reward)

        try:
            episode_metrics = []
            for episode_offset in range(int(episodes)):
                metrics = run_episode(
                    wrapped_env,
                    network,
                    device=self.device,
                    deterministic=True,
                    seed=int(seed_start) + episode_offset,
                    max_steps=self.evaluation_config.rollout.max_steps,
                    render=bool(render_mode == "human"),
                )
                episode_metrics.append(metrics.to_dict())
        finally:
            wrapped_env.close()

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


__all__ = ["PPOTraditionalTrainer"]

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from torch.distributions import Categorical

from ..core.runtime_config import load_runtime_config
from ..core.settings import load_stage_config
from ..core.types import EpisodeMetrics, RolloutBatch
from ..environment.config import init_env
from ..environment.reward import TraditionalRewardWrapper
from ..network.actor_critic import build_actor_critic


def _make_wrapped_env(
    *,
    config_path: str | Path | None,
    stage: str,
    render_mode: str | None,
):
    raw_config = load_runtime_config(config_path)
    stage_config = load_stage_config(stage, raw_config=raw_config)
    env = init_env(
        seed=0,
        stage=stage,
        config_path=config_path,
        render_mode=render_mode,
    )
    return TraditionalRewardWrapper(env, stage_config.reward)


def build_vector_env(
    *,
    config_path: str | Path | None = None,
    stage: str = "train",
    n_envs: int = 1,
    render_mode: str | None = None,
):
    env_fns = [
        partial(
            _make_wrapped_env,
            config_path=config_path,
            stage=stage,
            render_mode=render_mode,
        )
        for _ in range(int(n_envs))
    ]
    if int(n_envs) <= 1:
        return SyncVectorEnv(env_fns, autoreset_mode="SameStep")
    return AsyncVectorEnv(
        env_fns,
        shared_memory=False,
        context="spawn",
        autoreset_mode="SameStep",
    )


def compute_gae(
    *,
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    for index in range(rewards.shape[0] - 1, -1, -1):
        if index == rewards.shape[0] - 1:
            next_non_terminal = 1.0 - dones[index]
            next_value = next_values
        else:
            next_non_terminal = 1.0 - dones[index]
            next_value = values[index + 1]
        delta = rewards[index] + float(gamma) * next_value * next_non_terminal - values[index]
        last_advantage = (
            delta
            + float(gamma) * float(gae_lambda) * next_non_terminal * last_advantage
        )
        advantages[index] = last_advantage
    returns = advantages + values
    return advantages.astype(np.float32, copy=False), returns.astype(np.float32, copy=False)


def _prepare_batch_observation(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    obs_array = np.asarray(obs, dtype=np.float32)
    return torch.as_tensor(obs_array, dtype=torch.float32, device=device)


def _sample_actions(
    logits: torch.Tensor,
    *,
    deterministic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    distribution = Categorical(logits=logits)
    if deterministic:
        action_tensor = torch.argmax(logits, dim=1)
    else:
        action_tensor = distribution.sample()
    log_prob_tensor = distribution.log_prob(action_tensor)
    return (
        action_tensor.detach().cpu().numpy().astype(np.int64, copy=False),
        log_prob_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _extract_info_value(
    info_payload: dict | None,
    key: str,
    index: int,
    *,
    default,
):
    if not isinstance(info_payload, dict):
        return default
    mask = info_payload.get(f"_{key}")
    values = info_payload.get(key)
    if mask is not None:
        mask_array = np.asarray(mask)
        if mask_array.shape and not bool(mask_array[index]):
            return default
    if values is None:
        return default
    value_array = np.asarray(values, dtype=object)
    if value_array.shape:
        return value_array[index]
    return values


class VectorizedRolloutCollector:
    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        stage: str = "train",
        n_envs: int | None = None,
        seed_start: int = 21,
        render_mode: str | None = None,
    ) -> None:
        self.config_path = config_path
        self.raw_config = load_runtime_config(config_path)
        self.config = load_stage_config(stage, raw_config=self.raw_config)
        self.stage = stage
        self.n_envs = int(n_envs or self.config.rollout.n_envs)
        self.render_mode = render_mode
        self.vector_env = build_vector_env(
            config_path=config_path,
            stage=stage,
            n_envs=self.n_envs,
            render_mode=render_mode,
        )
        seeds = [int(seed_start) + offset for offset in range(self.n_envs)]
        self.current_obs, _ = self.vector_env.reset(seed=seeds)
        self.current_obs = np.asarray(self.current_obs, dtype=np.float32)
        self._running_fitness = np.zeros(self.n_envs, dtype=np.float32)
        self._running_lengths = np.zeros(self.n_envs, dtype=np.int32)

    def close(self) -> None:
        self.vector_env.close()

    def collect(
        self,
        network,
        *,
        device: str | torch.device = "cpu",
        steps_per_env: int | None = None,
        deterministic: bool = False,
    ) -> RolloutBatch:
        device_obj = torch.device(device)
        rollout_steps = int(steps_per_env or self.config.rollout.steps_per_env)
        obs_shape = tuple(int(axis) for axis in self.current_obs.shape[1:])

        obs_buffer = np.zeros((rollout_steps, self.n_envs, *obs_shape), dtype=np.float32)
        action_buffer = np.zeros((rollout_steps, self.n_envs), dtype=np.int64)
        log_prob_buffer = np.zeros((rollout_steps, self.n_envs), dtype=np.float32)
        reward_buffer = np.zeros((rollout_steps, self.n_envs), dtype=np.float32)
        done_buffer = np.zeros((rollout_steps, self.n_envs), dtype=np.bool_)
        value_buffer = np.zeros((rollout_steps, self.n_envs), dtype=np.float32)
        episode_metrics: list[EpisodeMetrics] = []

        for step_index in range(rollout_steps):
            obs_buffer[step_index] = self.current_obs
            obs_tensor = _prepare_batch_observation(self.current_obs, device_obj)
            with torch.no_grad():
                logits, values = network(obs_tensor, return_logits=True)
            actions, log_probs = _sample_actions(logits, deterministic=deterministic)

            next_obs, rewards, terminated, truncated, info = self.vector_env.step(actions)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=np.bool_)
            truncated = np.asarray(truncated, dtype=np.bool_)
            dones = np.logical_or(terminated, truncated)

            action_buffer[step_index] = actions
            log_prob_buffer[step_index] = log_probs
            reward_buffer[step_index] = rewards
            done_buffer[step_index] = dones
            value_buffer[step_index] = values.squeeze(-1).detach().cpu().numpy().astype(
                np.float32,
                copy=False,
            )

            self._running_fitness += rewards
            self._running_lengths += 1

            final_info = info.get("final_info")
            for env_index in np.nonzero(dones)[0]:
                metrics = EpisodeMetrics(
                    fitness=float(
                        _extract_info_value(
                            final_info,
                            "episode_fitness",
                            env_index,
                            default=self._running_fitness[env_index],
                        )
                    ),
                    collided=bool(
                        _extract_info_value(
                            final_info,
                            "collision",
                            env_index,
                            default=False,
                        )
                    ),
                    success=bool(
                        _extract_info_value(
                            final_info,
                            "success",
                            env_index,
                            default=False,
                        )
                    ),
                    distance_travelled=float(
                        _extract_info_value(
                            final_info,
                            "distance_travelled",
                            env_index,
                            default=0.0,
                        )
                    ),
                    episode_length=int(self._running_lengths[env_index]),
                    terminated=bool(terminated[env_index]),
                    truncated=bool(truncated[env_index]),
                )
                episode_metrics.append(metrics)
                self._running_fitness[env_index] = 0.0
                self._running_lengths[env_index] = 0

            self.current_obs = next_obs

        obs_tensor = _prepare_batch_observation(self.current_obs, device_obj)
        with torch.no_grad():
            _, next_values = network(obs_tensor, return_logits=True)
        next_values_array = next_values.squeeze(-1).detach().cpu().numpy().astype(
            np.float32,
            copy=False,
        )

        advantages, returns = compute_gae(
            rewards=reward_buffer,
            dones=done_buffer,
            values=value_buffer,
            next_values=next_values_array,
            gamma=float(self.config.ppo.gamma),
            gae_lambda=float(self.config.ppo.gae_lambda),
        )

        return RolloutBatch(
            obs=obs_buffer,
            actions=action_buffer,
            log_probs=log_prob_buffer,
            rewards=reward_buffer,
            dones=done_buffer,
            values=value_buffer,
            advantages=advantages,
            returns=returns,
            episode_metrics=tuple(episode_metrics),
        )


def run_episode(
    env,
    network,
    *,
    device: str | torch.device = "cpu",
    deterministic: bool,
    seed: int,
    max_steps: int | None = None,
    render: bool = False,
) -> EpisodeMetrics:
    device_obj = torch.device(device)
    obs, info = env.reset(seed=int(seed))
    terminated = False
    truncated = False
    episode_info = dict(info)
    step_count = 0

    while not (terminated or truncated):
        if max_steps is not None and step_count >= int(max_steps):
            truncated = True
            break

        obs_array = np.asarray(obs, dtype=np.float32)
        with torch.no_grad():
            logits, _ = network(
                torch.as_tensor(obs_array, dtype=torch.float32, device=device_obj).unsqueeze(0),
                return_logits=True,
            )
        action, _ = _sample_actions(logits, deterministic=deterministic)
        obs, _, terminated, truncated, episode_info = env.step(int(action[0]))
        if render:
            env.render()
        step_count += 1

    return EpisodeMetrics(
        fitness=float(episode_info.get("episode_fitness", 0.0)),
        collided=bool(episode_info.get("collision", episode_info.get("crashed", False))),
        success=bool(episode_info.get("success", False)),
        distance_travelled=float(episode_info.get("distance_travelled", 0.0)),
        episode_length=int(step_count),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def run_checkpoint_episode(
    policy_state_dict: dict[str, torch.Tensor],
    *,
    config_path: str | Path | None = None,
    stage: str = "evaluation",
    seed: int,
    deterministic: bool = True,
    render_mode: str | None = None,
    max_steps: int | None = None,
) -> EpisodeMetrics:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    raw_config = load_runtime_config(config_path)
    stage_config = load_stage_config(stage, raw_config=raw_config)
    env = init_env(
        seed=int(seed),
        stage=stage,
        config_path=config_path,
        render_mode=render_mode,
    )
    wrapped_env = TraditionalRewardWrapper(env, stage_config.reward)
    network = build_actor_critic(stage_config)
    network.load_state_dict(policy_state_dict)
    network.to(torch.device("cpu"))
    network.eval()

    try:
        return run_episode(
            wrapped_env,
            network,
            device="cpu",
            deterministic=bool(deterministic),
            seed=int(seed),
            max_steps=max_steps if max_steps is not None else stage_config.rollout.max_steps,
            render=bool(render_mode == "human"),
        )
    finally:
        wrapped_env.close()


__all__ = [
    "VectorizedRolloutCollector",
    "build_vector_env",
    "compute_gae",
    "run_checkpoint_episode",
    "run_episode",
]

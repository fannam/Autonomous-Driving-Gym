from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from torch.distributions import Categorical

from ..core.runtime_config import get_environment_config, load_runtime_config
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
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
    terminateds: np.ndarray | None = None,
    truncateds: np.ndarray | None = None,
    final_values: np.ndarray | None = None,
    dones: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized GAE that handles termination and truncation separately.

    - ``terminateds``: true termination (crash/offroad). V(s_next)=0.
    - ``truncateds``: time-limit truncation. Bootstraps from ``final_values[step]``.
    - ``final_values``: V(s_final) at each truncated step; ignored elsewhere.
    - ``dones``: legacy compatibility; if provided and ``terminateds`` is None, it is
      treated as terminated (old behaviour without truncation bootstrap).
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)

    if terminateds is None and truncateds is None:
        if dones is None:
            raise ValueError(
                "compute_gae requires either terminateds/truncateds or the legacy dones array."
            )
        terminateds_arr = np.asarray(dones, dtype=np.float32)
        truncateds_arr = np.zeros_like(terminateds_arr, dtype=np.float32)
    else:
        terminateds_arr = np.asarray(
            terminateds if terminateds is not None else np.zeros_like(rewards),
            dtype=np.float32,
        )
        truncateds_arr = np.asarray(
            truncateds if truncateds is not None else np.zeros_like(rewards),
            dtype=np.float32,
        )

    if final_values is None:
        final_values_arr = np.zeros_like(rewards, dtype=np.float32)
    else:
        final_values_arr = np.asarray(final_values, dtype=np.float32)

    done_mask = np.clip(terminateds_arr + truncateds_arr, 0.0, 1.0)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    for index in range(rewards.shape[0] - 1, -1, -1):
        if index == rewards.shape[0] - 1:
            next_value_chain = next_values
        else:
            next_value_chain = values[index + 1]

        # Boot-strap target:
        #   terminated → 0
        #   truncated  → V(final_obs)  (final_values)
        #   otherwise  → next_value_chain
        next_value = (
            (1.0 - done_mask[index]) * next_value_chain
            + truncateds_arr[index] * final_values_arr[index]
        )
        delta = rewards[index] + float(gamma) * next_value - values[index]
        last_advantage = (
            delta
            + float(gamma) * float(gae_lambda) * (1.0 - done_mask[index]) * last_advantage
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


def _resolve_action_labels(
    *,
    stage: str,
    raw_config: dict,
) -> tuple[str, ...]:
    environment = get_environment_config(stage=stage, raw_config=raw_config)
    env_config = environment.get("config", {})
    action_config = dict(env_config.get("action", {}) or {})
    if action_config.get("type") == "MultiAgentAction":
        action_config = dict(action_config.get("action_config", {}) or {})

    action_type = str(action_config.get("type", "DiscreteMetaAction"))
    longitudinal = bool(action_config.get("longitudinal", True))
    lateral = bool(action_config.get("lateral", True))

    if action_type == "DiscreteMetaAction":
        if longitudinal and lateral:
            return ("LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER")
        if longitudinal:
            return ("SLOWER", "IDLE", "FASTER")
        if lateral:
            return ("LANE_LEFT", "IDLE", "LANE_RIGHT")

    action_count = int(action_config.get("actions_per_axis", 5))
    if action_type == "DiscreteAction":
        axis_count = int(longitudinal) + int(lateral)
        action_count = int(action_count ** max(1, axis_count))
    return tuple(f"ACTION_{index}" for index in range(action_count))


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
        self.action_labels = _resolve_action_labels(
            stage=stage,
            raw_config=self.raw_config,
        )
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
        self._running_speed_sum = np.zeros(self.n_envs, dtype=np.float32)
        self._running_normalized_speed_sum = np.zeros(self.n_envs, dtype=np.float32)
        self._running_low_speed_ratio_sum = np.zeros(self.n_envs, dtype=np.float32)
        self._running_raw_env_reward_sum = np.zeros(self.n_envs, dtype=np.float32)
        self._running_offroad_steps = np.zeros(self.n_envs, dtype=np.int32)

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
        terminated_buffer = np.zeros((rollout_steps, self.n_envs), dtype=np.bool_)
        truncated_buffer = np.zeros((rollout_steps, self.n_envs), dtype=np.bool_)
        value_buffer = np.zeros((rollout_steps, self.n_envs), dtype=np.float32)
        final_value_buffer = np.zeros((rollout_steps, self.n_envs), dtype=np.float32)
        episode_metrics: list[EpisodeMetrics] = []
        batch_speed_sum = 0.0
        batch_normalized_speed_sum = 0.0
        batch_low_speed_ratio_sum = 0.0
        batch_raw_env_reward_sum = 0.0
        batch_offroad_steps = 0

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
            terminated_buffer[step_index] = terminated
            truncated_buffer[step_index] = truncated
            value_buffer[step_index] = values.squeeze(-1).detach().cpu().numpy().astype(
                np.float32,
                copy=False,
            )

            truncated_env_indices = np.nonzero(truncated)[0]
            if truncated_env_indices.size > 0:
                final_obs_array = info.get("final_obs")
                if final_obs_array is not None:
                    final_obs_batch = np.stack(
                        [
                            np.asarray(final_obs_array[env_idx], dtype=np.float32)
                            for env_idx in truncated_env_indices
                        ],
                        axis=0,
                    )
                    final_obs_tensor = torch.as_tensor(
                        final_obs_batch,
                        dtype=torch.float32,
                        device=device_obj,
                    )
                    with torch.no_grad():
                        _, final_values_tensor = network(
                            final_obs_tensor,
                            return_logits=True,
                        )
                    final_values_np = (
                        final_values_tensor.squeeze(-1)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    for local_idx, env_idx in enumerate(truncated_env_indices):
                        final_value_buffer[step_index, env_idx] = final_values_np[local_idx]

            self._running_fitness += rewards
            self._running_lengths += 1

            final_info = info.get("final_info")
            for env_index in range(self.n_envs):
                step_info_payload = final_info if bool(dones[env_index]) else info
                forward_speed = float(
                    _extract_info_value(
                        step_info_payload,
                        "forward_speed",
                        env_index,
                        default=0.0,
                    )
                )
                normalized_speed = float(
                    _extract_info_value(
                        step_info_payload,
                        "normalized_speed",
                        env_index,
                        default=0.0,
                    )
                )
                low_speed_ratio = float(
                    _extract_info_value(
                        step_info_payload,
                        "low_speed_ratio",
                        env_index,
                        default=0.0,
                    )
                )
                raw_env_reward = float(
                    _extract_info_value(
                        step_info_payload,
                        "raw_env_reward",
                        env_index,
                        default=0.0,
                    )
                )
                offroad = bool(
                    _extract_info_value(
                        step_info_payload,
                        "offroad",
                        env_index,
                        default=False,
                    )
                )

                self._running_speed_sum[env_index] += forward_speed
                self._running_normalized_speed_sum[env_index] += normalized_speed
                self._running_low_speed_ratio_sum[env_index] += low_speed_ratio
                self._running_raw_env_reward_sum[env_index] += raw_env_reward
                self._running_offroad_steps[env_index] += int(offroad)

                batch_speed_sum += forward_speed
                batch_normalized_speed_sum += normalized_speed
                batch_low_speed_ratio_sum += low_speed_ratio
                batch_raw_env_reward_sum += raw_env_reward
                batch_offroad_steps += int(offroad)

            for env_index in np.nonzero(dones)[0]:
                episode_length = max(1, int(self._running_lengths[env_index]))
                episode_length_float = float(episode_length)
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
                    episode_length=episode_length,
                    mean_speed_mps=float(self._running_speed_sum[env_index] / episode_length_float),
                    mean_speed_kph=float(self._running_speed_sum[env_index] * 3.6 / episode_length_float),
                    mean_normalized_speed=float(
                        self._running_normalized_speed_sum[env_index] / episode_length_float
                    ),
                    mean_low_speed_ratio=float(
                        self._running_low_speed_ratio_sum[env_index] / episode_length_float
                    ),
                    mean_step_reward=float(self._running_fitness[env_index] / episode_length_float),
                    mean_raw_env_reward=float(
                        self._running_raw_env_reward_sum[env_index] / episode_length_float
                    ),
                    offroad_rate=float(self._running_offroad_steps[env_index] / episode_length_float),
                    terminated=bool(terminated[env_index]),
                    truncated=bool(truncated[env_index]),
                )
                episode_metrics.append(metrics)
                self._running_fitness[env_index] = 0.0
                self._running_lengths[env_index] = 0
                self._running_speed_sum[env_index] = 0.0
                self._running_normalized_speed_sum[env_index] = 0.0
                self._running_low_speed_ratio_sum[env_index] = 0.0
                self._running_raw_env_reward_sum[env_index] = 0.0
                self._running_offroad_steps[env_index] = 0

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
            terminateds=terminated_buffer,
            truncateds=truncated_buffer,
            final_values=final_value_buffer,
            values=value_buffer,
            next_values=next_values_array,
            gamma=float(self.config.ppo.gamma),
            gae_lambda=float(self.config.ppo.gae_lambda),
        )

        total_samples = max(1, rollout_steps * self.n_envs)
        action_index_counts = np.bincount(
            action_buffer.reshape(-1),
            minlength=len(self.action_labels),
        )
        action_counts = {
            label: int(action_index_counts[index])
            for index, label in enumerate(self.action_labels)
        }
        action_fractions = {
            label: float(action_counts[label] / total_samples)
            for label in self.action_labels
        }
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
            mean_speed_mps=float(batch_speed_sum / total_samples),
            mean_speed_kph=float(batch_speed_sum * 3.6 / total_samples),
            mean_normalized_speed=float(batch_normalized_speed_sum / total_samples),
            mean_low_speed_ratio=float(batch_low_speed_ratio_sum / total_samples),
            mean_step_reward=float(np.mean(reward_buffer, dtype=np.float32)),
            mean_raw_env_reward=float(batch_raw_env_reward_sum / total_samples),
            offroad_rate=float(batch_offroad_steps / total_samples),
            finished_episode_count=int(len(episode_metrics)),
            action_counts=action_counts,
            action_fractions=action_fractions,
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
    speed_sum_mps = 0.0
    normalized_speed_sum = 0.0
    low_speed_ratio_sum = 0.0
    raw_env_reward_sum = 0.0
    shaped_reward_sum = 0.0
    offroad_steps = 0

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
        obs, reward, terminated, truncated, episode_info = env.step(int(action[0]))
        if render:
            env.render()
        shaped_reward_sum += float(reward)
        speed_sum_mps += float(episode_info.get("forward_speed", 0.0))
        normalized_speed_sum += float(episode_info.get("normalized_speed", 0.0))
        low_speed_ratio_sum += float(episode_info.get("low_speed_ratio", 0.0))
        raw_env_reward_sum += float(episode_info.get("raw_env_reward", 0.0))
        offroad_steps += int(bool(episode_info.get("offroad", False)))
        step_count += 1

    step_count_float = float(max(1, step_count))
    return EpisodeMetrics(
        fitness=float(episode_info.get("episode_fitness", 0.0)),
        collided=bool(episode_info.get("collision", episode_info.get("crashed", False))),
        success=bool(episode_info.get("success", False)),
        distance_travelled=float(episode_info.get("distance_travelled", 0.0)),
        episode_length=int(step_count),
        mean_speed_mps=float(speed_sum_mps / step_count_float),
        mean_speed_kph=float(speed_sum_mps * 3.6 / step_count_float),
        mean_normalized_speed=float(normalized_speed_sum / step_count_float),
        mean_low_speed_ratio=float(low_speed_ratio_sum / step_count_float),
        mean_step_reward=float(shaped_reward_sum / step_count_float),
        mean_raw_env_reward=float(raw_env_reward_sum / step_count_float),
        offroad_rate=float(offroad_steps / step_count_float),
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

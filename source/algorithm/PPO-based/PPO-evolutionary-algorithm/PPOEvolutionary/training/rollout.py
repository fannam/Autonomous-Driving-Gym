from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from ..core.runtime_config import load_runtime_config
from ..core.settings import load_stage_config
from ..core.types import EpisodeMetrics, TrajectoryBatch
from ..environment.config import init_env
from ..environment.reward import EvolutionaryRewardWrapper
from ..network.actor_critic import build_actor_critic


def _prepare_observation(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    obs_array = np.asarray(obs, dtype=np.float32)
    return torch.as_tensor(obs_array, dtype=torch.float32, device=device).unsqueeze(0)


def _select_action(
    logits: torch.Tensor,
    *,
    deterministic: bool,
) -> tuple[int, float]:
    flat_logits = logits.squeeze(0)
    distribution = Categorical(logits=flat_logits)
    if deterministic:
        action_tensor = torch.argmax(flat_logits, dim=0)
    else:
        action_tensor = distribution.sample()
    log_prob = distribution.log_prob(action_tensor)
    return int(action_tensor.item()), float(log_prob.item())


def run_episode(
    env,
    network,
    *,
    device: str | torch.device = "cpu",
    deterministic: bool,
    policy_index: int,
    seed: int,
    max_steps: int | None = None,
    render: bool = False,
) -> TrajectoryBatch:
    device_obj = torch.device(device)
    obs, info = env.reset(seed=int(seed))
    obs_buffer: list[np.ndarray] = []
    actions: list[int] = []
    behavior_log_probs: list[float] = []
    rewards: list[float] = []
    dones: list[bool] = []

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
            logits, _ = network(_prepare_observation(obs_array, device_obj), return_logits=True)
        action, log_prob = _select_action(logits, deterministic=deterministic)

        next_obs, reward, terminated, truncated, episode_info = env.step(action)
        if render:
            env.render()

        obs_buffer.append(obs_array)
        actions.append(int(action))
        behavior_log_probs.append(float(log_prob))
        rewards.append(float(reward))
        dones.append(bool(terminated or truncated))

        obs = next_obs
        step_count += 1

    last_obs = np.asarray(obs, dtype=np.float32)
    metrics = EpisodeMetrics(
        fitness=float(episode_info.get("episode_fitness", np.sum(rewards, dtype=np.float32))),
        collided=bool(episode_info.get("collision", episode_info.get("crashed", False))),
        success=bool(episode_info.get("success", False)),
        distance_travelled=float(episode_info.get("distance_travelled", 0.0)),
        episode_length=int(step_count),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )

    obs_shape = last_obs.shape
    empty_obs = np.zeros((0, *obs_shape), dtype=np.float32)
    return TrajectoryBatch(
        obs=np.stack(obs_buffer, axis=0).astype(np.float32, copy=False) if obs_buffer else empty_obs,
        actions=np.asarray(actions, dtype=np.int64),
        behavior_log_probs=np.asarray(behavior_log_probs, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        last_obs=last_obs.astype(np.float32, copy=False),
        episode_metrics=metrics,
        policy_index=int(policy_index),
        seed=int(seed),
    )


def run_rollout_episode(
    policy_state_dict: dict[str, torch.Tensor],
    *,
    config_path: str | Path | None = None,
    stage: str = "train",
    seed: int,
    policy_index: int,
    deterministic: bool = False,
    render_mode: str | None = None,
    max_steps: int | None = None,
) -> TrajectoryBatch:
    torch.set_num_threads(1)
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
    wrapped_env = EvolutionaryRewardWrapper(env, stage_config.reward)
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
            policy_index=int(policy_index),
            seed=int(seed),
            max_steps=max_steps if max_steps is not None else stage_config.rollout.max_steps,
            render=bool(render_mode == "human"),
        )
    finally:
        wrapped_env.close()


__all__ = ["run_episode", "run_rollout_episode"]

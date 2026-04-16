from __future__ import annotations

import numpy as np
import torch

from PPOEvolutionary.core.settings import TRAIN_CONFIG
from PPOEvolutionary.network.actor_critic import build_actor_critic
from PPOEvolutionary.training.rollout import run_rollout_episode
from tools.repo_layout import PPO_EVOLUTIONARY_ROOT


CONFIG_PATH = (
    PPO_EVOLUTIONARY_ROOT / "configs" / "highway_ppo_evolutionary.yaml"
).resolve()


def _build_policy_state():
    torch.manual_seed(0)
    return build_actor_critic(TRAIN_CONFIG).state_dict()


def test_rollout_worker_returns_expected_schema() -> None:
    trajectory = run_rollout_episode(
        _build_policy_state(),
        config_path=CONFIG_PATH,
        stage="train",
        seed=23,
        policy_index=1,
        deterministic=False,
        max_steps=3,
    )

    assert trajectory.policy_index == 1
    assert trajectory.seed == 23
    assert trajectory.obs.ndim == 4
    assert trajectory.obs.shape[1:] == TRAIN_CONFIG.observation_shape
    assert trajectory.last_obs.shape == TRAIN_CONFIG.observation_shape
    assert trajectory.actions.ndim == 1
    assert trajectory.behavior_log_probs.ndim == 1
    assert trajectory.rewards.ndim == 1
    assert trajectory.dones.ndim == 1
    assert len(trajectory.actions) == len(trajectory.behavior_log_probs)
    assert len(trajectory.actions) == len(trajectory.rewards)
    assert len(trajectory.actions) == len(trajectory.dones)
    assert np.isfinite(trajectory.behavior_log_probs).all()


def test_rollout_worker_is_seed_deterministic() -> None:
    state_dict = _build_policy_state()
    trajectory_a = run_rollout_episode(
        state_dict,
        config_path=CONFIG_PATH,
        stage="train",
        seed=31,
        policy_index=0,
        deterministic=False,
        max_steps=3,
    )
    trajectory_b = run_rollout_episode(
        state_dict,
        config_path=CONFIG_PATH,
        stage="train",
        seed=31,
        policy_index=0,
        deterministic=False,
        max_steps=3,
    )

    assert np.array_equal(trajectory_a.actions, trajectory_b.actions)
    assert np.allclose(trajectory_a.rewards, trajectory_b.rewards)
    assert np.allclose(trajectory_a.behavior_log_probs, trajectory_b.behavior_log_probs)

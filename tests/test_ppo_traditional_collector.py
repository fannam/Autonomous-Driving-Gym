from __future__ import annotations

import numpy as np
import torch
import yaml

from PPOTraditional.core.settings import TRAIN_CONFIG
from PPOTraditional.network.actor_critic import build_actor_critic
from PPOTraditional.training.collector import VectorizedRolloutCollector
from tools.repo_layout import PPO_TRADITIONAL_ROOT


CONFIG_PATH = (
    PPO_TRADITIONAL_ROOT / "configs" / "highway_ppo_traditional.yaml"
).resolve()


def _build_network():
    torch.manual_seed(0)
    network = build_actor_critic(TRAIN_CONFIG)
    network.eval()
    return network


def _make_short_horizon_config(tmp_path):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["environment"]["config"]["duration"] = 3
    config["environment"]["config"]["vehicles_count"] = 8
    config["environment"]["config"]["vehicles_density"] = 1.0

    config_path = tmp_path / "highway_ppo_traditional_collector_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_collector_returns_expected_vectorized_schema() -> None:
    collector = VectorizedRolloutCollector(
        config_path=CONFIG_PATH,
        stage="train",
        n_envs=2,
        seed_start=23,
    )
    try:
        batch = collector.collect(
            _build_network(),
            device="cpu",
            steps_per_env=3,
            deterministic=False,
        )
    finally:
        collector.close()

    assert batch.obs.shape == (3, 2, *TRAIN_CONFIG.observation_shape)
    assert batch.actions.shape == (3, 2)
    assert batch.log_probs.shape == (3, 2)
    assert batch.rewards.shape == (3, 2)
    assert batch.dones.shape == (3, 2)
    assert batch.values.shape == (3, 2)
    assert batch.advantages.shape == (3, 2)
    assert batch.returns.shape == (3, 2)
    assert np.isfinite(batch.log_probs).all()
    assert np.isfinite(batch.values).all()
    assert tuple(batch.action_counts.keys()) == (
        "LANE_LEFT",
        "IDLE",
        "LANE_RIGHT",
        "FASTER",
        "SLOWER",
    )
    assert sum(batch.action_counts.values()) == batch.actions.size
    assert abs(sum(batch.action_fractions.values()) - 1.0) < 1e-6


def test_collector_is_seed_deterministic_in_evaluation_mode() -> None:
    network = _build_network()

    collector_a = VectorizedRolloutCollector(
        config_path=CONFIG_PATH,
        stage="train",
        n_envs=2,
        seed_start=31,
    )
    collector_b = VectorizedRolloutCollector(
        config_path=CONFIG_PATH,
        stage="train",
        n_envs=2,
        seed_start=31,
    )
    try:
        batch_a = collector_a.collect(
            network,
            device="cpu",
            steps_per_env=3,
            deterministic=True,
        )
        batch_b = collector_b.collect(
            network,
            device="cpu",
            steps_per_env=3,
            deterministic=True,
        )
    finally:
        collector_a.close()
        collector_b.close()

    assert np.array_equal(batch_a.actions, batch_b.actions)
    assert np.allclose(batch_a.log_probs, batch_b.log_probs)
    assert np.allclose(batch_a.rewards, batch_b.rewards)


def test_collector_records_episode_metrics_when_env_autoresets(tmp_path) -> None:
    config_path = _make_short_horizon_config(tmp_path)
    collector = VectorizedRolloutCollector(
        config_path=config_path,
        stage="train",
        n_envs=2,
        seed_start=47,
    )
    try:
        batch = collector.collect(
            _build_network(),
            device="cpu",
            steps_per_env=12,
            deterministic=True,
        )
    finally:
        collector.close()

    assert batch.episode_metrics
    assert all(metrics.episode_length > 0 for metrics in batch.episode_metrics)
    assert all(np.isfinite(metrics.fitness) for metrics in batch.episode_metrics)

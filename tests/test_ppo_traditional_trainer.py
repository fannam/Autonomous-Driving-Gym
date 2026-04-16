from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from PPOTraditional.training.collector import compute_gae
from PPOTraditional.training.trainer import PPOTraditionalTrainer
from tools.repo_layout import PPO_TRADITIONAL_ROOT


def _make_smoke_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    base_config_path = (
        PPO_TRADITIONAL_ROOT / "configs" / "highway_ppo_traditional.yaml"
    ).resolve()
    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))

    checkpoint_path = tmp_path / "ppo_traditional_smoke.pt"
    metrics_path = tmp_path / "ppo_traditional_metrics.jsonl"

    config["environment"]["config"]["duration"] = 3
    config["environment"]["config"]["vehicles_count"] = 8
    config["environment"]["config"]["vehicles_density"] = 1.0
    config["presets"]["train"]["rollout"]["n_envs"] = 2
    config["presets"]["train"]["rollout"]["steps_per_env"] = 3
    config["presets"]["train"]["rollout"]["max_steps"] = 3
    config["presets"]["train"]["ppo"]["minibatch_size"] = 4
    config["presets"]["train"]["logging"]["metrics_path"] = str(metrics_path)
    config["presets"]["train"]["model_path"] = str(checkpoint_path)
    config["presets"]["evaluation"]["rollout"]["max_steps"] = 3
    config["presets"]["evaluation"]["logging"]["metrics_path"] = str(metrics_path)
    config["presets"]["evaluation"]["model_path"] = str(checkpoint_path)

    smoke_config_path = tmp_path / "highway_ppo_traditional_smoke.yaml"
    smoke_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return smoke_config_path, checkpoint_path, metrics_path


def test_compute_gae_keeps_advantages_partitioned_per_env() -> None:
    rewards = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dones = np.asarray([[False, True], [False, False]], dtype=np.bool_)
    values = np.asarray([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    next_values = np.asarray([2.5, 3.0], dtype=np.float32)

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_values=next_values,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert np.isclose(
        advantages[0, 1],
        rewards[0, 1] - values[0, 1],
    )
    assert np.isfinite(advantages).all()
    assert np.isfinite(returns).all()


@pytest.fixture(scope="module")
def trained_smoke_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("ppo_traditional_smoke")
    config_path, checkpoint_path, metrics_path = _make_smoke_config(tmp_path)
    trainer = PPOTraditionalTrainer(
        config_path=config_path,
        device="cpu",
        verbose=False,
    )
    summaries = trainer.fit(
        updates=1,
        n_envs=2,
        steps_per_env=3,
        seed_start=17,
    )
    return {
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "summaries": summaries,
    }


def test_trainer_smoke_fit_writes_checkpoint_and_metrics(trained_smoke_run) -> None:
    checkpoint_path = trained_smoke_run["checkpoint_path"]
    metrics_path = trained_smoke_run["metrics_path"]
    summaries = trained_smoke_run["summaries"]

    assert checkpoint_path.exists()
    assert metrics_path.exists()
    assert metrics_path.read_text(encoding="utf-8").strip()
    assert len(summaries) == 1
    assert "best_fitness" in summaries[0]
    assert "policy_loss" in summaries[0]
    assert "total_timesteps" in summaries[0]


def test_evaluation_smoke_loads_checkpoint_and_reports_metrics(trained_smoke_run) -> None:
    trainer = PPOTraditionalTrainer(
        config_path=trained_smoke_run["config_path"],
        device="cpu",
        verbose=False,
    )
    summary = trainer.evaluate(
        checkpoint_path=trained_smoke_run["checkpoint_path"],
        policy_source="best",
        episodes=1,
        seed_start=101,
    )

    assert summary["policy_source"] == "best"
    assert len(summary["episodes"]) == 1
    assert "mean_fitness" in summary
    assert "mean_distance" in summary
    assert "collision_rate" in summary
    assert "success_rate" in summary

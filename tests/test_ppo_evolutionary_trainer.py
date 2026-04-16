from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from PPOEvolutionary.training.trainer import PPOEvolutionaryTrainer
from tools.repo_layout import PPO_EVOLUTIONARY_ROOT


def _make_smoke_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    base_config_path = (
        PPO_EVOLUTIONARY_ROOT / "configs" / "highway_ppo_evolutionary.yaml"
    ).resolve()
    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))

    checkpoint_path = tmp_path / "ppo_evolutionary_smoke.pt"
    metrics_path = tmp_path / "ppo_evolutionary_metrics.jsonl"

    config["environment"]["config"]["duration"] = 3
    config["environment"]["config"]["vehicles_count"] = 8
    config["environment"]["config"]["vehicles_density"] = 1.0
    config["presets"]["train"]["rollout"]["workers"] = 1
    config["presets"]["train"]["rollout"]["max_steps"] = 3
    config["presets"]["train"]["ppo"]["minibatch_size"] = 4
    config["presets"]["train"]["logging"]["metrics_path"] = str(metrics_path)
    config["presets"]["train"]["model_path"] = str(checkpoint_path)
    config["presets"]["evaluation"]["rollout"]["max_steps"] = 3
    config["presets"]["evaluation"]["logging"]["metrics_path"] = str(metrics_path)
    config["presets"]["evaluation"]["model_path"] = str(checkpoint_path)

    smoke_config_path = tmp_path / "highway_ppo_evolutionary_smoke.yaml"
    smoke_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return smoke_config_path, checkpoint_path, metrics_path


@pytest.fixture(scope="module")
def trained_smoke_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("ppo_evolutionary_smoke")
    config_path, checkpoint_path, metrics_path = _make_smoke_config(tmp_path)
    trainer = PPOEvolutionaryTrainer(
        config_path=config_path,
        device="cpu",
        verbose=False,
    )
    summaries = trainer.fit(
        generations=1,
        population_size=2,
        workers=1,
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
    assert "mean_speed_kph" in summaries[0]
    assert "mean_episode_length" in summaries[0]
    assert "mean_step_reward" in summaries[0]
    assert "collision_rate" in summaries[0]


def test_evaluation_smoke_loads_checkpoint_and_reports_metrics(trained_smoke_run) -> None:
    trainer = PPOEvolutionaryTrainer(
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

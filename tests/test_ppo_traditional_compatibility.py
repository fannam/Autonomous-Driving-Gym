from __future__ import annotations

from PPOTraditional.core.types import UpdateSummary
from PPOTraditional.training.trainer import PPOTraditionalTrainer


def test_update_summary_uses_shared_metric_field_names() -> None:
    summary = UpdateSummary(
        update=1,
        total_timesteps=128,
        best_fitness=1.0,
        mean_fitness=0.5,
        collision_rate=0.25,
        success_rate=0.75,
        mean_distance=10.0,
        mean_episode_length=5.0,
        policy_loss=0.1,
        value_loss=0.2,
        entropy=0.3,
        approx_kl=0.01,
    ).to_dict()

    assert {
        "best_fitness",
        "mean_fitness",
        "collision_rate",
        "success_rate",
        "mean_distance",
        "mean_episode_length",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
    }.issubset(summary.keys())


def test_trainer_module_does_not_depend_on_ppo_evolutionary() -> None:
    module_name = PPOTraditionalTrainer.__module__
    assert "PPOEvolutionary" not in module_name

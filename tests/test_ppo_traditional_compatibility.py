from __future__ import annotations

from PPOTraditional.core.types import UpdateSummary
from PPOTraditional.training.trainer import PPOTraditionalTrainer


def test_update_summary_uses_shared_metric_field_names() -> None:
    summary = UpdateSummary(
        update=1,
        total_timesteps=128,
        best_fitness=1.0,
        mean_fitness=0.5,
        ema_fitness=0.5,
        best_ema_fitness=0.5,
        learning_rate=3e-4,
        collision_rate=0.25,
        success_rate=0.75,
        mean_distance=10.0,
        mean_episode_length=5.0,
        mean_speed_mps=8.0,
        mean_speed_kph=28.8,
        mean_normalized_speed=0.4,
        mean_low_speed_ratio=0.1,
        mean_step_reward=0.2,
        mean_raw_env_reward=0.0,
        offroad_rate=0.0,
        finished_episode_count=2,
        sample_count=128,
        policy_loss=0.1,
        value_loss=0.2,
        entropy=0.3,
        approx_kl=0.01,
        action_counts={"LANE_LEFT": 1, "IDLE": 2, "LANE_RIGHT": 3, "FASTER": 4, "SLOWER": 5},
        action_fractions={
            "LANE_LEFT": 0.1,
            "IDLE": 0.2,
            "LANE_RIGHT": 0.3,
            "FASTER": 0.2,
            "SLOWER": 0.2,
        },
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
        "action_counts",
        "action_fractions",
    }.issubset(summary.keys())


def test_trainer_module_does_not_depend_on_ppo_evolutionary() -> None:
    module_name = PPOTraditionalTrainer.__module__
    assert "PPOEvolutionary" not in module_name

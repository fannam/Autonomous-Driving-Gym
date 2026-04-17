from __future__ import annotations

from PPOTraditional.core.runtime_config import get_scenario_config_path
from PPOTraditional.core.settings import EVALUATION_CONFIG, TRAIN_CONFIG
from PPOTraditional.environment.config import build_env_spec
from tools.repo_layout import PPO_TRADITIONAL_ROOT


def test_ppo_traditional_runtime_config_finds_package_layout() -> None:
    expected = (
        PPO_TRADITIONAL_ROOT / "configs" / "highway_ppo_traditional.yaml"
    ).resolve()
    config_path = get_scenario_config_path("highway_ppo_traditional")

    assert config_path == expected


def test_ppo_traditional_settings_infer_observation_and_action_shape() -> None:
    assert TRAIN_CONFIG.observation_shape == (4, 100, 24)
    assert TRAIN_CONFIG.n_actions == 5
    assert TRAIN_CONFIG.network.residual_blocks == 10
    assert EVALUATION_CONFIG.observation_shape == (4, 100, 24)
    assert EVALUATION_CONFIG.n_actions == 5


def test_ppo_traditional_stage_env_overrides_success_termination() -> None:
    config_path = (
        PPO_TRADITIONAL_ROOT / "configs" / "highway_ppo_traditional.yaml"
    ).resolve()

    train_spec = build_env_spec(stage="train", config_path=config_path)
    evaluation_spec = build_env_spec(stage="evaluation", config_path=config_path)

    assert train_spec.config["terminate_on_success"] is False
    assert evaluation_spec.config["terminate_on_success"] is True
    assert train_spec.config["action"]["type"] == "DiscreteMetaAction"
    assert train_spec.config["action"]["target_speeds"] == [18, 22, 26, 30, 34, 38]

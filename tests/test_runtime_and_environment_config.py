from __future__ import annotations

from AlphaZeroAdversarial.core.runtime_config import (
    resolve_scenario_config_path as resolve_adversarial_config_path,
)
from AlphaZeroAdversarial.environment.config import (
    build_env_spec as build_adversarial_env_spec,
)
from AlphaZeroMetaAdversarial.core.runtime_config import (
    resolve_scenario_config_path as resolve_meta_config_path,
)
from AlphaZeroMetaAdversarial.environment.config import (
    build_env_spec as build_meta_env_spec,
)
from tools.repo_layout import (
    ALPHAZERO_ADVERSARIAL_ROOT,
    ALPHAZERO_META_ADVERSARIAL_ROOT,
)


def test_adversarial_runtime_config_resolves_moved_layout_paths() -> None:
    expected = (
        ALPHAZERO_ADVERSARIAL_ROOT / "configs" / "racetrack_adversarial.yaml"
    ).resolve()
    resolved = resolve_adversarial_config_path("racetrack_adversarial")

    assert resolved == expected


def test_meta_runtime_config_resolves_moved_layout_paths() -> None:
    expected = (
        ALPHAZERO_META_ADVERSARIAL_ROOT / "configs" / "highway_meta_adversarial.yaml"
    ).resolve()
    resolved = resolve_meta_config_path("highway_meta_adversarial")

    assert resolved == expected


def test_build_env_spec_merges_overrides_for_adversarial_config() -> None:
    config_path = (
        ALPHAZERO_ADVERSARIAL_ROOT / "configs" / "racetrack_adversarial.yaml"
    ).resolve()

    spec = build_adversarial_env_spec(
        stage="self_play",
        config_path=config_path,
        env_config_overrides={
            "duration": 17,
            "action": {
                "action_config": {
                    "longitudinal": False,
                }
            },
        },
    )

    assert spec.scenario_name == "racetrack_adversarial"
    assert spec.env_id == "racetrack-v0"
    assert spec.render_mode == "rgb_array"
    assert spec.config["duration"] == 17
    assert spec.config["action"]["type"] == "MultiAgentAction"
    assert spec.config["action"]["action_config"]["longitudinal"] is False
    assert spec.config["action"]["action_config"]["lateral"] is True


def test_build_env_spec_accepts_config_path_for_meta_adversarial() -> None:
    config_path = (
        ALPHAZERO_META_ADVERSARIAL_ROOT / "configs" / "highway_meta_adversarial.yaml"
    ).resolve()

    spec = build_meta_env_spec(
        stage="evaluation",
        config_path=config_path,
        env_config_overrides={
            "duration": 23,
            "action": {
                "action_config": {
                    "target_speeds": [18, 24, 30],
                }
            },
        },
    )

    assert spec.scenario_name == "highway_meta_adversarial"
    assert spec.env_id == "highway-v0"
    assert spec.render_mode == "rgb_array"
    assert spec.config["duration"] == 23
    assert spec.config["action"]["action_config"]["type"] == "DiscreteMetaAction"
    assert spec.config["action"]["action_config"]["target_speeds"] == [18, 24, 30]

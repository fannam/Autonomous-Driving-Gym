from __future__ import annotations

import numpy as np

from AlphaZeroAdversarial.core.runtime_config import (
    get_scenario_config_path as get_adversarial_config_path,
)
from AlphaZeroAdversarial.environment.config import (
    build_env_spec as build_adversarial_env_spec,
)
from AlphaZeroMetaAdversarial.core.runtime_config import (
    get_scenario_config_path as get_meta_config_path,
)
from AlphaZeroMetaAdversarial.core.settings import SELF_PLAY_CONFIG as META_SELF_PLAY_CONFIG
from AlphaZeroMetaAdversarial.environment.config import (
    build_env_spec as build_meta_env_spec,
    init_env as init_meta_env,
)
from tools.repo_layout import (
    ALPHAZERO_ADVERSARIAL_ROOT,
    ALPHAZERO_META_ADVERSARIAL_ROOT,
)


def test_adversarial_runtime_config_finds_moved_layout_paths() -> None:
    expected = (
        ALPHAZERO_ADVERSARIAL_ROOT / "configs" / "racetrack_adversarial.yaml"
    ).resolve()
    config_path = get_adversarial_config_path("racetrack_adversarial")

    assert config_path == expected


def test_meta_runtime_config_finds_moved_layout_paths() -> None:
    expected = (
        ALPHAZERO_META_ADVERSARIAL_ROOT / "configs" / "highway_meta_adversarial.yaml"
    ).resolve()
    config_path = get_meta_config_path("highway_meta_adversarial")

    assert config_path == expected


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
            "road_speed_limit": 44,
            "action": {
                "action_config": {
                    "target_speeds": [18, 22, 26, 30, 34, 38],
                }
            },
        },
    )

    assert spec.scenario_name == "highway_meta_adversarial"
    assert spec.env_id == "highway-v0"
    assert spec.render_mode == "rgb_array"
    assert spec.config["duration"] == 23
    assert spec.config["road_speed_limit"] == 44
    assert spec.config["action"]["action_config"]["type"] == "DiscreteMetaAction"
    assert spec.config["action"]["action_config"]["target_speeds"] == [18, 22, 26, 30, 34, 38]
    assert spec.config["action"]["agents_action_config_overrides"] == [
        None,
        {"target_speeds": [22, 26, 30, 34, 38, 42]},
    ]


def test_meta_env_applies_configurable_road_speed_limit() -> None:
    env = init_meta_env(
        stage="self_play",
        env_config_overrides={
            "duration": 1,
            "road_speed_limit": 44,
        },
    )
    try:
        graph = env.unwrapped.road.network.graph
        start_node = next(iter(graph))
        end_node = next(iter(graph[start_node]))
        lane = graph[start_node][end_node][0]
        assert float(lane.speed_limit) == 44.0
    finally:
        env.close()


def test_meta_env_uses_default_asymmetric_target_speeds() -> None:
    env = init_meta_env(
        stage="self_play",
        env_config_overrides={
            "duration": 1,
        },
    )
    try:
        ego_vehicle, npc_vehicle = env.unwrapped.controlled_vehicles[:2]
        action_type = env.unwrapped.action_type

        assert np.allclose(
            ego_vehicle.target_speeds,
            np.asarray([18, 22, 26, 30, 34, 38], dtype=np.float32),
        )
        assert np.allclose(
            npc_vehicle.target_speeds,
            np.asarray([22, 26, 30, 34, 38, 42], dtype=np.float32),
        )
        assert np.allclose(
            action_type.agents_action_types[0].target_speeds,
            np.asarray([18, 22, 26, 30, 34, 38], dtype=np.float32),
        )
        assert np.allclose(
            action_type.agents_action_types[1].target_speeds,
            np.asarray([22, 26, 30, 34, 38, 42], dtype=np.float32),
        )
    finally:
        env.close()


def test_meta_self_play_config_enables_discounted_values_and_npc_removal() -> None:
    assert abs(float(META_SELF_PLAY_CONFIG.discount_gamma) - 0.99) < 1e-9
    assert META_SELF_PLAY_CONFIG.zero_sum.remove_npc_on_self_fault is True
    assert int(META_SELF_PLAY_CONFIG.n_actions) == 5
    assert float(META_SELF_PLAY_CONFIG.npc_closing_ucb_bonus) == 0.15

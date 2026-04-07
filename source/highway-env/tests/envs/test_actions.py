import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import highway_env


gym.register_envs(highway_env)


@pytest.mark.parametrize(
    "action_config",
    [
        {"type": "ContinuousAction"},
        {"type": "DiscreteAction"},
        {"type": "DiscreteMetaAction"},
    ],
)
def test_action_type(action_config):
    env = gym.make("highway-v0", config={"action": action_config})
    env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.action_space.contains(action)
        assert env.observation_space.contains(obs)
    env.close()


def _multi_agent_action_config(
    *,
    agents_action_config_overrides=None,
):
    action = {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": True,
            "target_speeds": [18, 22, 26, 30, 34, 38],
        },
    }
    if agents_action_config_overrides is not None:
        action["agents_action_config_overrides"] = agents_action_config_overrides
    return {
        "controlled_vehicles": 2,
        "vehicles_count": 6,
        "action": action,
    }


def test_multi_agent_action_without_per_agent_overrides_reuses_base_action_config():
    env = gym.make("highway-v0", config=_multi_agent_action_config())
    try:
        env.reset()
        controlled_vehicles = env.unwrapped.controlled_vehicles
        action_type = env.unwrapped.action_type

        assert len(controlled_vehicles) == 2
        np.testing.assert_allclose(
            controlled_vehicles[0].target_speeds,
            np.asarray([18, 22, 26, 30, 34, 38], dtype=np.float32),
        )
        np.testing.assert_allclose(
            controlled_vehicles[1].target_speeds,
            np.asarray([18, 22, 26, 30, 34, 38], dtype=np.float32),
        )
        np.testing.assert_allclose(
            action_type.agents_action_types[0].target_speeds,
            np.asarray([18, 22, 26, 30, 34, 38], dtype=np.float32),
        )
        np.testing.assert_allclose(
            action_type.agents_action_types[1].target_speeds,
            np.asarray([18, 22, 26, 30, 34, 38], dtype=np.float32),
        )
    finally:
        env.close()


def test_multi_agent_action_supports_per_agent_target_speed_overrides():
    env = gym.make(
        "highway-v0",
        config=_multi_agent_action_config(
            agents_action_config_overrides=[
                None,
                {"target_speeds": [22, 26, 30, 34, 38, 42]},
            ]
        ),
    )
    try:
        env.reset()
        controlled_vehicles = env.unwrapped.controlled_vehicles
        action_type = env.unwrapped.action_type

        assert len(controlled_vehicles) == 2
        np.testing.assert_allclose(
            controlled_vehicles[0].target_speeds,
            np.asarray([18, 22, 26, 30, 34, 38], dtype=np.float32),
        )
        np.testing.assert_allclose(
            controlled_vehicles[1].target_speeds,
            np.asarray([22, 26, 30, 34, 38, 42], dtype=np.float32),
        )
        np.testing.assert_allclose(
            action_type.agents_action_types[0].target_speeds,
            np.asarray([18, 22, 26, 30, 34, 38], dtype=np.float32),
        )
        np.testing.assert_allclose(
            action_type.agents_action_types[1].target_speeds,
            np.asarray([22, 26, 30, 34, 38, 42], dtype=np.float32),
        )
        assert isinstance(env.action_space, spaces.Tuple)
        assert len(env.action_space.spaces) == 2
        assert [space.n for space in env.action_space.spaces] == [5, 5]
        assert env.action_space.contains((1, 1))
    finally:
        env.close()


def test_multi_agent_action_validates_override_count():
    with pytest.raises(
        ValueError,
        match="agents_action_config_overrides must align with controlled_vehicles",
    ):
        gym.make(
            "highway-v0",
            config=_multi_agent_action_config(
                agents_action_config_overrides=[None],
            ),
        )


@pytest.mark.parametrize(
    "override",
    (
        {"type": "DiscreteAction", "actions_per_axis": 3},
        {"longitudinal": False},
    ),
    ids=("action_type", "cardinality"),
)
def test_multi_agent_action_rejects_incompatible_agent_overrides(override):
    with pytest.raises(
        ValueError,
        match="agents_action_config_overrides must preserve the base action",
    ):
        gym.make(
            "highway-v0",
            config=_multi_agent_action_config(
                agents_action_config_overrides=[None, override],
            ),
        )

from __future__ import annotations

import numpy as np

from PPOEvolutionary.core.settings import TRAIN_CONFIG
from PPOEvolutionary.environment.config import init_env
from PPOEvolutionary.environment.reward import EvolutionaryRewardWrapper


def _make_wrapped_env(**env_overrides):
    env = init_env(
        stage="train",
        env_config_overrides={"duration": 5, **env_overrides},
    )
    return EvolutionaryRewardWrapper(env, TRAIN_CONFIG.reward)


def test_reward_wrapper_normalizes_speed_from_target_range() -> None:
    wrapped_env = _make_wrapped_env()
    try:
        wrapped_env.reset(seed=3)
        vehicle = wrapped_env.unwrapped.vehicle
        vehicle.heading = 0.0

        vehicle.speed = 18.0
        assert abs(wrapped_env.compute_reward_terms()["normalized_speed"] - 0.0) < 1e-6

        vehicle.speed = 28.0
        assert abs(wrapped_env.compute_reward_terms()["normalized_speed"] - 0.5) < 1e-6

        vehicle.speed = 38.0
        assert abs(wrapped_env.compute_reward_terms()["normalized_speed"] - 1.0) < 1e-6
    finally:
        wrapped_env.close()


def test_reward_wrapper_uses_lane_position_and_penalties() -> None:
    wrapped_env = _make_wrapped_env()
    try:
        wrapped_env.reset(seed=5)
        vehicle = wrapped_env.unwrapped.vehicle
        from_node, to_node, _ = vehicle.lane_index

        vehicle.lane_index = (from_node, to_node, 3)
        vehicle.target_lane_index = vehicle.lane_index
        vehicle.crashed = True
        vehicle.position = np.asarray([vehicle.position[0], 1000.0], dtype=np.float32)

        terms = wrapped_env.compute_reward_terms()
        shaped_reward = wrapped_env.compute_shaped_reward(terms)

        assert abs(terms["right_lane_score"] - 1.0) < 1e-6
        assert terms["collision"] == 1.0
        assert terms["offroad"] == 1.0
        assert shaped_reward <= -9.0
    finally:
        wrapped_env.close()


def test_reward_wrapper_tracks_episode_fitness_across_steps() -> None:
    wrapped_env = _make_wrapped_env(vehicles_count=0, vehicles_density=1.0)
    try:
        wrapped_env.reset(seed=11)
        _, reward1, terminated1, truncated1, info1 = wrapped_env.step(1)
        assert not (terminated1 or truncated1)

        _, reward2, _, _, info2 = wrapped_env.step(1)

        assert np.isfinite(reward1)
        assert np.isfinite(reward2)
        assert abs((reward1 + reward2) - info2["episode_fitness"]) < 1e-6
        assert "raw_env_reward" in info1
        assert "reward_terms" in info2
    finally:
        wrapped_env.close()

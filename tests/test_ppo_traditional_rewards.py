from __future__ import annotations

from dataclasses import replace

import numpy as np

from PPOTraditional.core.settings import TRAIN_CONFIG
from PPOTraditional.environment.config import init_env
from PPOTraditional.environment.reward import TraditionalRewardWrapper


def _make_wrapped_env(**env_overrides):
    env = init_env(
        stage="train",
        env_config_overrides={"duration": 5, **env_overrides},
    )
    return TraditionalRewardWrapper(env, TRAIN_CONFIG.reward)


def _make_overtake_env(*, overtake_reward: float = 1.0, **env_overrides):
    env = init_env(
        stage="train",
        env_config_overrides={"duration": 5, **env_overrides},
    )
    reward_config = replace(
        TRAIN_CONFIG.reward,
        overtake_reward=float(overtake_reward),
        overtake_range=60.0,
        overtake_clearance=0.0,
        collision_penalty=0.0,
        offroad_penalty=0.0,
        speed_weight=0.0,
        low_speed_penalty=0.0,
        lane_change_reward=0.0,
    )
    return TraditionalRewardWrapper(env, reward_config)


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


def test_reward_wrapper_penalizes_low_speed() -> None:
    wrapped_env = _make_wrapped_env()
    try:
        wrapped_env.reset(seed=5)
        vehicle = wrapped_env.unwrapped.vehicle
        min_speed, _ = wrapped_env._get_speed_bounds()
        threshold_speed = (
            float(TRAIN_CONFIG.reward.low_speed_threshold_multiplier) * float(min_speed)
        )

        vehicle.heading = 0.0
        vehicle.speed = float(min_speed)

        slow_terms = wrapped_env.compute_reward_terms()
        slow_reward = wrapped_env.compute_shaped_reward(slow_terms)

        vehicle.speed = float(threshold_speed)
        threshold_terms = wrapped_env.compute_reward_terms()
        threshold_reward = wrapped_env.compute_shaped_reward(threshold_terms)

        assert abs(slow_terms["low_speed_ratio"] - 1.0) < 1e-6
        assert abs(slow_reward + float(TRAIN_CONFIG.reward.low_speed_penalty)) < 1e-6
        assert abs(threshold_terms["low_speed_ratio"] - 0.0) < 1e-6
        assert threshold_reward > slow_reward
    finally:
        wrapped_env.close()


def test_reward_wrapper_applies_collision_and_offroad_penalties() -> None:
    wrapped_env = _make_wrapped_env()
    try:
        wrapped_env.reset(seed=7)
        vehicle = wrapped_env.unwrapped.vehicle
        vehicle.heading = 0.0
        vehicle.speed = 38.0
        vehicle.crashed = True
        vehicle.position = np.asarray([vehicle.position[0], 1000.0], dtype=np.float32)

        terms = wrapped_env.compute_reward_terms()
        shaped_reward = wrapped_env.compute_shaped_reward(terms)
        expected_reward = (
            float(TRAIN_CONFIG.reward.speed_weight)
            - float(TRAIN_CONFIG.reward.collision_penalty)
            - float(TRAIN_CONFIG.reward.offroad_penalty)
        )

        assert abs(terms["low_speed_ratio"] - 0.0) < 1e-6
        assert terms["collision"] == 1.0
        assert terms["offroad"] == 1.0
        assert abs(shaped_reward - expected_reward) < 1e-6
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
        assert "low_speed_penalty" in info2["reward_terms"]
    finally:
        wrapped_env.close()


def _spawn_vehicle_at(wrapped_env, longitudinal_offset: float, lateral: float | None = None):
    """Create and register a vehicle on the same road as ego at the given offset."""
    from highway_env.vehicle.kinematics import Vehicle as HwVehicle

    ego = wrapped_env.unwrapped.vehicle
    lateral_position = float(ego.position[1]) if lateral is None else float(lateral)
    position = np.asarray(
        [float(ego.position[0]) + float(longitudinal_offset), lateral_position],
        dtype=np.float32,
    )
    vehicle = HwVehicle(
        wrapped_env.unwrapped.road,
        position=position,
        heading=0.0,
        speed=0.0,
    )
    wrapped_env.unwrapped.road.vehicles.append(vehicle)
    return vehicle


def test_overtake_counts_when_ego_passes_vehicle_within_range() -> None:
    wrapped = _make_overtake_env(vehicles_count=0, vehicles_density=1.0)
    try:
        wrapped.reset(seed=1)
        ego = wrapped.unwrapped.vehicle
        ego.heading = 0.0

        other = _spawn_vehicle_at(wrapped, longitudinal_offset=30.0)
        wrapped._prev_ahead_snapshot = wrapped._snapshot_vehicles_ahead()
        assert id(other) in wrapped._prev_ahead_snapshot

        ego.position = np.asarray(
            [float(other.position[0]) + 1.0, float(ego.position[1])],
            dtype=np.float32,
        )

        assert wrapped._count_overtakes() == 1
    finally:
        wrapped.close()


def test_overtake_not_credited_when_vehicle_despawns() -> None:
    wrapped = _make_overtake_env(vehicles_count=0, vehicles_density=1.0)
    try:
        wrapped.reset(seed=2)
        ego = wrapped.unwrapped.vehicle
        ego.heading = 0.0

        other = _spawn_vehicle_at(wrapped, longitudinal_offset=25.0)
        wrapped._prev_ahead_snapshot = wrapped._snapshot_vehicles_ahead()
        assert id(other) in wrapped._prev_ahead_snapshot

        wrapped.unwrapped.road.vehicles.remove(other)

        assert wrapped._count_overtakes() == 0
    finally:
        wrapped.close()


def test_overtake_not_counted_for_vehicle_beyond_range() -> None:
    wrapped = _make_overtake_env(vehicles_count=0, vehicles_density=1.0)
    try:
        wrapped.reset(seed=3)
        ego = wrapped.unwrapped.vehicle
        ego.heading = 0.0

        _spawn_vehicle_at(wrapped, longitudinal_offset=120.0)
        snapshot = wrapped._snapshot_vehicles_ahead()
        assert snapshot == {}

        wrapped._prev_ahead_snapshot = snapshot
        assert wrapped._count_overtakes() == 0
    finally:
        wrapped.close()


def test_overtake_not_credited_on_collision_step() -> None:
    wrapped = _make_overtake_env(vehicles_count=0, vehicles_density=1.0)
    try:
        wrapped.reset(seed=4)
        ego = wrapped.unwrapped.vehicle
        ego.heading = 0.0

        other = _spawn_vehicle_at(wrapped, longitudinal_offset=20.0)
        wrapped._prev_ahead_snapshot = wrapped._snapshot_vehicles_ahead()
        assert id(other) in wrapped._prev_ahead_snapshot

        ego.position = np.asarray(
            [float(other.position[0]) + 1.0, float(ego.position[1])],
            dtype=np.float32,
        )
        ego.crashed = True

        assert wrapped._count_overtakes() == 0
    finally:
        wrapped.close()


def test_overtake_not_credited_when_offroad() -> None:
    wrapped = _make_overtake_env(vehicles_count=0, vehicles_density=1.0)
    try:
        wrapped.reset(seed=5)
        ego = wrapped.unwrapped.vehicle
        ego.heading = 0.0

        other = _spawn_vehicle_at(wrapped, longitudinal_offset=15.0)
        wrapped._prev_ahead_snapshot = wrapped._snapshot_vehicles_ahead()

        ego.position = np.asarray(
            [float(other.position[0]) + 1.0, 500.0],
            dtype=np.float32,
        )
        # Highway env sets ``on_road`` dynamically based on position; force it
        # to avoid relying on geometry.
        type(ego).on_road = property(lambda self: False)  # type: ignore[assignment]
        try:
            assert wrapped._count_overtakes() == 0
        finally:
            # Restore the property so other tests are unaffected.
            del type(ego).on_road
    finally:
        wrapped.close()


def test_overtake_short_circuits_when_reward_zero() -> None:
    wrapped = _make_overtake_env(
        overtake_reward=0.0,
        vehicles_count=0,
        vehicles_density=1.0,
    )
    try:
        wrapped.reset(seed=6)
        ego = wrapped.unwrapped.vehicle
        ego.heading = 0.0

        other = _spawn_vehicle_at(wrapped, longitudinal_offset=20.0)
        wrapped._prev_ahead_snapshot = {id(other): float(other.position[0])}

        ego.position = np.asarray(
            [float(other.position[0]) + 1.0, float(ego.position[1])],
            dtype=np.float32,
        )
        assert wrapped._count_overtakes() == 0
    finally:
        wrapped.close()


def test_overtake_respects_clearance_threshold() -> None:
    wrapped = _make_overtake_env(vehicles_count=0, vehicles_density=1.0)
    wrapped.reward_config = replace(
        wrapped.reward_config,
        overtake_clearance=5.0,
    )
    try:
        wrapped.reset(seed=7)
        ego = wrapped.unwrapped.vehicle
        ego.heading = 0.0

        other = _spawn_vehicle_at(wrapped, longitudinal_offset=20.0)
        wrapped._prev_ahead_snapshot = wrapped._snapshot_vehicles_ahead()

        # Move ego just 1m past the other vehicle — below the 5m clearance.
        ego.position = np.asarray(
            [float(other.position[0]) + 1.0, float(ego.position[1])],
            dtype=np.float32,
        )
        assert wrapped._count_overtakes() == 0

        # Move ego 6m past — clearance satisfied.
        ego.position = np.asarray(
            [float(other.position[0]) + 6.0, float(ego.position[1])],
            dtype=np.float32,
        )
        assert wrapped._count_overtakes() == 1
    finally:
        wrapped.close()


def test_overtake_counts_multiple_simultaneous_passes() -> None:
    wrapped = _make_overtake_env(vehicles_count=0, vehicles_density=1.0)
    try:
        wrapped.reset(seed=8)
        ego = wrapped.unwrapped.vehicle
        ego.heading = 0.0

        first = _spawn_vehicle_at(wrapped, longitudinal_offset=15.0)
        second = _spawn_vehicle_at(wrapped, longitudinal_offset=25.0)
        third = _spawn_vehicle_at(wrapped, longitudinal_offset=40.0)

        snapshot = wrapped._snapshot_vehicles_ahead()
        assert {id(first), id(second), id(third)}.issubset(snapshot.keys())
        wrapped._prev_ahead_snapshot = snapshot

        # Ego jumps past all three vehicles.
        ego.position = np.asarray(
            [float(third.position[0]) + 2.0, float(ego.position[1])],
            dtype=np.float32,
        )
        assert wrapped._count_overtakes() == 3
    finally:
        wrapped.close()


def test_overtake_not_double_counted_after_snapshot_refresh() -> None:
    wrapped = _make_overtake_env(vehicles_count=0, vehicles_density=1.0)
    try:
        wrapped.reset(seed=9)
        ego = wrapped.unwrapped.vehicle
        ego.heading = 0.0

        other = _spawn_vehicle_at(wrapped, longitudinal_offset=20.0)
        wrapped._prev_ahead_snapshot = wrapped._snapshot_vehicles_ahead()

        ego.position = np.asarray(
            [float(other.position[0]) + 2.0, float(ego.position[1])],
            dtype=np.float32,
        )

        terms_first = wrapped.compute_reward_terms()
        assert float(terms_first["overtake_count"]) == 1.0

        # Next step: ego still ahead of the vehicle — no new overtake.
        ego.position = np.asarray(
            [float(ego.position[0]) + 2.0, float(ego.position[1])],
            dtype=np.float32,
        )
        terms_second = wrapped.compute_reward_terms()
        assert float(terms_second["overtake_count"]) == 0.0
    finally:
        wrapped.close()


def test_overtake_reward_included_in_shaped_reward() -> None:
    wrapped = _make_overtake_env(
        overtake_reward=1.0,
        vehicles_count=0,
        vehicles_density=1.0,
    )
    try:
        wrapped.reset(seed=10)
        ego = wrapped.unwrapped.vehicle
        ego.heading = 0.0

        other = _spawn_vehicle_at(wrapped, longitudinal_offset=20.0)
        wrapped._prev_ahead_snapshot = wrapped._snapshot_vehicles_ahead()

        ego.position = np.asarray(
            [float(other.position[0]) + 2.0, float(ego.position[1])],
            dtype=np.float32,
        )

        terms = wrapped.compute_reward_terms()
        shaped = wrapped.compute_shaped_reward(terms)
        assert abs(shaped - 1.0) < 1e-6
        assert int(terms["overtake_count"]) == 1
    finally:
        wrapped.close()

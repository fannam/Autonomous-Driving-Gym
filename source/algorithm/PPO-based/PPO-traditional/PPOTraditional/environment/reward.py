from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from highway_env.vehicle.kinematics import Vehicle

from ..core.settings import RewardConfig


class TraditionalRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        reward_config: RewardConfig | dict[str, Any],
    ) -> None:
        super().__init__(env)
        self.reward_config = (
            reward_config
            if isinstance(reward_config, RewardConfig)
            else RewardConfig.from_dict(reward_config)
        )
        self.episode_fitness = 0.0
        self._prev_lane_index = None
        self._steps_since_lane_change = 0
        self._prev_ahead_snapshot: dict[int, float] = {}
        self._prev_ego_longitudinal: float | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_fitness = 0.0
        vehicle = self.unwrapped.vehicle
        self._prev_lane_index = getattr(vehicle, "lane_index", None) if vehicle else None
        self._steps_since_lane_change = 0
        self._prev_ahead_snapshot = self._snapshot_vehicles_ahead()
        self._prev_ego_longitudinal = self._ego_longitudinal_position()
        return obs, dict(info)

    def _action_config(self) -> dict[str, Any]:
        action_config = dict(getattr(self.unwrapped, "config", {}).get("action", {}) or {})
        if action_config.get("type") == "MultiAgentAction":
            action_config = dict(action_config.get("action_config", {}) or {})
        return action_config

    def _get_speed_bounds(self) -> tuple[float, float]:
        reward_speed_range = getattr(self.unwrapped, "config", {}).get("reward_speed_range")
        if reward_speed_range is not None:
            speeds = np.asarray(reward_speed_range, dtype=np.float32).reshape(-1)
            if speeds.size >= 2:
                return float(np.min(speeds[:2])), float(np.max(speeds[:2]))
        action_config = self._action_config()
        target_speeds = action_config.get("target_speeds")
        if target_speeds is not None:
            speeds = np.asarray(target_speeds, dtype=np.float32).reshape(-1)
            if speeds.size >= 2:
                return float(np.min(speeds)), float(np.max(speeds))
        return max(0.0, float(Vehicle.MIN_SPEED)), max(0.0, float(Vehicle.MAX_SPEED))

    def _forward_speed(self) -> float:
        vehicle = self.unwrapped.vehicle
        return float(vehicle.speed) * float(np.cos(float(vehicle.heading)))

    def _normalized_speed(self) -> float:
        min_speed, max_speed = self._get_speed_bounds()
        span = max(1e-6, max_speed - min_speed)
        forward_speed = self._forward_speed()
        return float(np.clip((forward_speed - min_speed) / span, 0.0, 1.0))

    def _low_speed_ratio(self) -> float:
        min_speed, _ = self._get_speed_bounds()
        threshold_speed = float(self.reward_config.low_speed_threshold_multiplier) * float(min_speed)
        threshold_speed = max(float(min_speed) + 1e-6, threshold_speed)
        deficit = max(0.0, threshold_speed - self._forward_speed())
        normalizer = max(1e-6, threshold_speed - float(min_speed))
        return float(np.clip(deficit / normalizer, 0.0, 1.0))

    # ----- Overtake detection -----

    def _longitudinal_position(self, vehicle) -> float:
        """Longitudinal coordinate on the vehicle's lane, with safe fallbacks.

        Falls back to ``position[0]`` if lane lookup fails (e.g. the vehicle is
        between lanes mid-change or the road network is unavailable).
        """
        if vehicle is None:
            return 0.0
        try:
            road = getattr(self.unwrapped, "road", None)
            if road is None or getattr(road, "network", None) is None:
                raise AttributeError
            lane_index = getattr(vehicle, "lane_index", None)
            if lane_index is None:
                raise AttributeError
            lane = road.network.get_lane(lane_index)
            longitudinal, _ = lane.local_coordinates(vehicle.position)
            value = float(longitudinal)
            if not np.isfinite(value):
                raise ValueError
            return value
        except Exception:
            try:
                return float(vehicle.position[0])
            except Exception:
                return 0.0

    def _ego_longitudinal_position(self) -> float | None:
        vehicle = getattr(self.unwrapped, "vehicle", None)
        if vehicle is None:
            return None
        return self._longitudinal_position(vehicle)

    def _iter_other_vehicles(self):
        road = getattr(self.unwrapped, "road", None)
        if road is None:
            return
        ego = self.unwrapped.vehicle
        for vehicle in list(getattr(road, "vehicles", []) or []):
            if vehicle is ego or vehicle is None:
                continue
            yield vehicle

    def _snapshot_vehicles_ahead(self) -> dict[int, float]:
        """Return ``{id(vehicle): longitudinal_position}`` for every vehicle that
        is strictly ahead of the ego within the configured range.

        The snapshot is taken on the unwrapped env's current state, so it must be
        called either right after ``reset`` or at the end of a ``step`` (after
        the physics have been updated).
        """
        ego = getattr(self.unwrapped, "vehicle", None)
        if ego is None:
            return {}
        range_limit = float(getattr(self.reward_config, "overtake_range", 60.0))
        if range_limit <= 0.0:
            return {}
        ego_lon = self._longitudinal_position(ego)
        snapshot: dict[int, float] = {}
        for vehicle in self._iter_other_vehicles():
            if not bool(getattr(vehicle, "on_road", True)):
                continue
            lon = self._longitudinal_position(vehicle)
            if not np.isfinite(lon):
                continue
            dx = lon - ego_lon
            if 0.0 < dx <= range_limit:
                snapshot[id(vehicle)] = lon
        return snapshot

    def _count_overtakes(self) -> int:
        """Count vehicles that were ahead of ego last step and are at / behind it now.

        Edge cases handled:
          * ego crashed or off-road in the current step → no overtake credit
          * vehicle despawned / off-road in the current step → not counted
          * ego went backwards: handled implicitly — we compare current positions
            so a vehicle only counts if it is *actually* no longer ahead
          * ``overtake_reward`` is zero or ``overtake_range`` is non-positive →
            short-circuit to zero (no work, no reward)
          * ``overtake_clearance`` allows requiring ego to be cleanly past the
            vehicle (``current_ego_lon - current_vehicle_lon >= clearance``)
          * id() reuse by a newly spawned vehicle → mitigated by requiring the
            id to appear in the current ``road.vehicles`` list; if a stale id is
            reused we still require the position to satisfy the overtake check.
        """
        if float(getattr(self.reward_config, "overtake_reward", 0.0)) == 0.0:
            return 0
        if float(getattr(self.reward_config, "overtake_range", 0.0)) <= 0.0:
            return 0
        if not self._prev_ahead_snapshot:
            return 0

        ego = getattr(self.unwrapped, "vehicle", None)
        if ego is None:
            return 0
        if bool(getattr(ego, "crashed", False)):
            return 0
        if not bool(getattr(ego, "on_road", True)):
            return 0

        ego_lon = self._longitudinal_position(ego)
        if not np.isfinite(ego_lon):
            return 0

        clearance = max(0.0, float(getattr(self.reward_config, "overtake_clearance", 0.0)))

        current_by_id: dict[int, float] = {}
        for vehicle in self._iter_other_vehicles():
            if not bool(getattr(vehicle, "on_road", True)):
                continue
            lon = self._longitudinal_position(vehicle)
            if not np.isfinite(lon):
                continue
            current_by_id[id(vehicle)] = lon

        overtake_count = 0
        for vehicle_id in self._prev_ahead_snapshot:
            current_lon = current_by_id.get(vehicle_id)
            if current_lon is None:
                # Vehicle no longer in the scene (despawn, off-road, etc.).
                # Don't credit this as an overtake.
                continue
            if (ego_lon - current_lon) >= clearance:
                overtake_count += 1
        return overtake_count

    # ----- Reward terms -----

    def compute_reward_terms(self) -> dict[str, float]:
        vehicle = self.unwrapped.vehicle
        current_lane_index = getattr(vehicle, "lane_index", None) if vehicle else None
        lane_changed = 0.0
        lane_change_cooldown = getattr(self.reward_config, "lane_change_cooldown", 10)

        # Detect lane change
        if self._prev_lane_index is not None and current_lane_index is not None:
            if (
                len(self._prev_lane_index) >= 3
                and len(current_lane_index) >= 3
                and self._prev_lane_index[2] != current_lane_index[2]
            ):
                # Only reward if cooldown period has passed (prevent zigzagging)
                if self._steps_since_lane_change >= lane_change_cooldown:
                    lane_changed = 1.0
                self._steps_since_lane_change = 0

        self._prev_lane_index = current_lane_index
        self._steps_since_lane_change += 1

        overtake_count = self._count_overtakes()
        # Refresh snapshot and ego position for the next step.
        self._prev_ahead_snapshot = self._snapshot_vehicles_ahead()
        self._prev_ego_longitudinal = self._ego_longitudinal_position()

        return {
            "forward_speed": self._forward_speed(),
            "normalized_speed": self._normalized_speed(),
            "low_speed_ratio": self._low_speed_ratio(),
            "collision": float(bool(getattr(vehicle, "crashed", False))),
            "offroad": float(not bool(getattr(vehicle, "on_road", True))),
            "lane_changed": lane_changed,
            "overtake_count": float(overtake_count),
        }

    def compute_shaped_reward(self, terms: dict[str, float]) -> float:
        lane_change_reward = getattr(self.reward_config, "lane_change_reward", 0.0)
        overtake_reward = getattr(self.reward_config, "overtake_reward", 0.0)
        return (
            float(self.reward_config.speed_weight) * float(terms["normalized_speed"])
            - float(self.reward_config.low_speed_penalty) * float(terms["low_speed_ratio"])
            - float(self.reward_config.collision_penalty) * float(terms["collision"])
            - float(self.reward_config.offroad_penalty) * float(terms["offroad"])
            + float(lane_change_reward) * float(terms.get("lane_changed", 0.0))
            + float(overtake_reward) * float(terms.get("overtake_count", 0.0))
        )

    def step(self, action):
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        terms = self.compute_reward_terms()
        shaped_reward = self.compute_shaped_reward(terms)
        self.episode_fitness += shaped_reward

        lane_change_reward = getattr(self.reward_config, "lane_change_reward", 0.0)
        overtake_reward = getattr(self.reward_config, "overtake_reward", 0.0)
        reward_terms = {
            "speed_reward": float(self.reward_config.speed_weight) * float(terms["normalized_speed"]),
            "low_speed_penalty": -float(self.reward_config.low_speed_penalty)
            * float(terms["low_speed_ratio"]),
            "collision_penalty": -float(self.reward_config.collision_penalty)
            * float(terms["collision"]),
            "offroad_penalty": -float(self.reward_config.offroad_penalty) * float(terms["offroad"]),
            "lane_change_reward": float(lane_change_reward) * float(terms.get("lane_changed", 0.0)),
            "overtake_reward": float(overtake_reward) * float(terms.get("overtake_count", 0.0)),
        }
        augmented_info = dict(info)
        augmented_info.update(
            {
                "reward_terms": reward_terms,
                "forward_speed": float(terms["forward_speed"]),
                "normalized_speed": float(terms["normalized_speed"]),
                "low_speed_ratio": float(terms["low_speed_ratio"]),
                "collision": bool(terms["collision"]),
                "offroad": bool(terms["offroad"]),
                "lane_changed": bool(terms.get("lane_changed", False)),
                "overtake_count": int(terms.get("overtake_count", 0.0)),
                "raw_env_reward": float(raw_reward),
                "shaped_reward": float(shaped_reward),
                "distance_travelled": float(info.get("distance_travelled", 0.0)),
                "success": bool(info.get("success", False)),
                "episode_fitness": float(self.episode_fitness),
            }
        )
        return obs, shaped_reward, terminated, truncated, augmented_info


__all__ = ["TraditionalRewardWrapper"]

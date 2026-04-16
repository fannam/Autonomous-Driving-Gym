from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from highway_env.vehicle.controller import ControlledVehicle
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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_fitness = 0.0
        return obs, dict(info)

    def _action_config(self) -> dict[str, Any]:
        action_config = dict(getattr(self.unwrapped, "config", {}).get("action", {}) or {})
        if action_config.get("type") == "MultiAgentAction":
            action_config = dict(action_config.get("action_config", {}) or {})
        return action_config

    def _get_speed_bounds(self) -> tuple[float, float]:
        action_config = self._action_config()
        target_speeds = action_config.get("target_speeds")
        if target_speeds is not None:
            speeds = np.asarray(target_speeds, dtype=np.float32).reshape(-1)
            if speeds.size >= 2:
                return float(np.min(speeds)), float(np.max(speeds))
        reward_speed_range = getattr(self.unwrapped, "config", {}).get("reward_speed_range")
        if reward_speed_range is not None:
            speeds = np.asarray(reward_speed_range, dtype=np.float32).reshape(-1)
            if speeds.size >= 2:
                return float(np.min(speeds[:2])), float(np.max(speeds[:2]))
        return max(0.0, float(Vehicle.MIN_SPEED)), max(0.0, float(Vehicle.MAX_SPEED))

    def _normalized_speed(self) -> float:
        vehicle = self.unwrapped.vehicle
        min_speed, max_speed = self._get_speed_bounds()
        span = max(1e-6, max_speed - min_speed)
        forward_speed = float(vehicle.speed) * float(np.cos(float(vehicle.heading)))
        return float(np.clip((forward_speed - min_speed) / span, 0.0, 1.0))

    def _right_lane_score(self) -> float:
        vehicle = self.unwrapped.vehicle
        road = getattr(self.unwrapped, "road", None)
        if road is None:
            return 0.0
        neighbours = road.network.all_side_lanes(vehicle.lane_index)
        lane_index = (
            vehicle.target_lane_index[2]
            if isinstance(vehicle, ControlledVehicle)
            else vehicle.lane_index[2]
        )
        return float(lane_index / max(len(neighbours) - 1, 1))

    def compute_reward_terms(self) -> dict[str, float]:
        vehicle = self.unwrapped.vehicle
        return {
            "normalized_speed": self._normalized_speed(),
            "right_lane_score": self._right_lane_score(),
            "collision": float(bool(getattr(vehicle, "crashed", False))),
            "offroad": float(not bool(getattr(vehicle, "on_road", True))),
        }

    def compute_shaped_reward(self, terms: dict[str, float]) -> float:
        return (
            float(self.reward_config.speed_weight) * float(terms["normalized_speed"])
            + float(self.reward_config.right_lane_weight) * float(terms["right_lane_score"])
            - float(self.reward_config.collision_penalty) * float(terms["collision"])
            - float(self.reward_config.offroad_penalty) * float(terms["offroad"])
        )

    def step(self, action):
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        terms = self.compute_reward_terms()
        shaped_reward = self.compute_shaped_reward(terms)
        self.episode_fitness += shaped_reward

        reward_terms = {
            "speed_reward": float(self.reward_config.speed_weight) * float(terms["normalized_speed"]),
            "right_lane_reward": float(self.reward_config.right_lane_weight)
            * float(terms["right_lane_score"]),
            "collision_penalty": -float(self.reward_config.collision_penalty)
            * float(terms["collision"]),
            "offroad_penalty": -float(self.reward_config.offroad_penalty) * float(terms["offroad"]),
        }
        augmented_info = dict(info)
        augmented_info.update(
            {
                "reward_terms": reward_terms,
                "normalized_speed": float(terms["normalized_speed"]),
                "right_lane_score": float(terms["right_lane_score"]),
                "collision": bool(terms["collision"]),
                "offroad": bool(terms["offroad"]),
                "raw_env_reward": float(raw_reward),
                "shaped_reward": float(shaped_reward),
                "distance_travelled": float(info.get("distance_travelled", 0.0)),
                "success": bool(info.get("success", False)),
                "episode_fitness": float(self.episode_fitness),
            }
        )
        return obs, shaped_reward, terminated, truncated, augmented_info


__all__ = ["TraditionalRewardWrapper"]

from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
                "success_mode": None,
                "terminate_on_success": False,
                "success_max_steps": None,
                "success_min_distance": None,
                "success_speed_range_fraction": 0.75,
                "success_speed_max_fraction": 0.8,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self._reset_success_state()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25.0,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _success_mode(self) -> str | None:
        raw_mode = self.config.get("success_mode")
        if raw_mode in (None, False, "", "none"):
            return None
        mode = str(raw_mode)
        if mode != "distance":
            raise ValueError(
                f"Unsupported success_mode={mode!r}. Only 'distance' is supported."
            )
        return mode

    def _get_success_speed_bounds(self) -> tuple[float, float]:
        action_config = self.config.get("action", {})
        target_speeds = action_config.get("target_speeds")
        if target_speeds is None:
            target_speeds = getattr(self.vehicle, "target_speeds", None)
        if target_speeds is not None:
            speeds = np.asarray(target_speeds, dtype=np.float32).reshape(-1)
            if speeds.size:
                return float(np.min(speeds)), float(np.max(speeds))

        reward_speed_range = self.config.get("reward_speed_range")
        if reward_speed_range is not None:
            speeds = np.asarray(reward_speed_range, dtype=np.float32).reshape(-1)
            if speeds.size >= 2:
                low = float(np.min(speeds[:2]))
                high = float(np.max(speeds[:2]))
                return low, high

        return max(0.0, float(Vehicle.MIN_SPEED)), max(0.0, float(Vehicle.MAX_SPEED))

    def _compute_required_success_distance(self) -> float | None:
        if self._success_mode() != "distance":
            return None

        explicit_threshold = self.config.get("success_min_distance")
        if explicit_threshold is not None:
            return max(0.0, float(explicit_threshold))

        min_speed, max_speed = self._get_success_speed_bounds()
        min_speed = max(0.0, float(min_speed))
        max_speed = max(min_speed, float(max_speed))

        range_fraction = float(self.config.get("success_speed_range_fraction", 0.75))
        max_fraction = float(self.config.get("success_speed_max_fraction", 0.8))
        required_avg_speed = max(
            min_speed + range_fraction * (max_speed - min_speed),
            max_fraction * max_speed,
        )

        policy_frequency = max(1e-6, float(self.config["policy_frequency"]))
        configured_max_steps = self.config.get("success_max_steps")
        if configured_max_steps is None:
            max_steps = float(self.config["duration"]) * policy_frequency
        else:
            max_steps = float(configured_max_steps)
        return max_steps * required_avg_speed / policy_frequency

    def _get_longitudinal_progress(self) -> float:
        try:
            lane = self.road.network.get_lane(self.vehicle.lane_index)
            longitudinal, _ = lane.local_coordinates(self.vehicle.position)
            return float(longitudinal)
        except Exception:
            return float(self.vehicle.position[0])

    def _reset_success_state(self) -> None:
        self._success_initial_progress = self._get_longitudinal_progress()
        self._required_success_distance = self._compute_required_success_distance()

    def _distance_travelled(self) -> float:
        return max(0.0, self._get_longitudinal_progress() - self._success_initial_progress)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            (
                bool(self.config.get("terminate_on_success", False))
                and self._is_success()
            )
            or self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _is_success(self) -> bool:
        if self._success_mode() != "distance":
            return False
        if bool(getattr(self.vehicle, "crashed", False)) or not bool(self.vehicle.on_road):
            return False
        if self._required_success_distance is None:
            return False
        return self._distance_travelled() >= self._required_success_distance

    def _info(self, obs: Observation, action: Action | None = None) -> dict:
        info = super()._info(obs, action)
        if self._success_mode() == "distance":
            info.update(
                {
                    "success": bool(self._is_success()),
                    "distance_travelled": float(self._distance_travelled()),
                    "required_success_distance": float(
                        self._required_success_distance or 0.0
                    ),
                }
            )
        return info


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False

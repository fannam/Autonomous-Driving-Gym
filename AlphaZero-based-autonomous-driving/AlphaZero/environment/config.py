import gymnasium as gym
import highway_env
import numpy as np


DEFAULT_OBSERVATION_FEATURES = ["presence", "on_lane", "on_road"]
DEFAULT_GRID_SIZE = [[-50, 50], [-12, 12]]
DEFAULT_GRID_STEP = [1.0, 1.0]


def compute_grid_shape(grid_size, grid_step):
    grid_size_arr = np.asarray(grid_size, dtype=np.float32)
    grid_step_arr = np.asarray(grid_step, dtype=np.float32)
    grid_shape = np.floor((grid_size_arr[:, 1] - grid_size_arr[:, 0]) / grid_step_arr)
    return tuple(int(axis_cells) for axis_cells in grid_shape)


class EnvironmentFactory:
    @staticmethod
    def default_config(vehicle_density=1):
        return {
            "observation": {
                "type": "DetailedOccupancyGrid",
                "vehicles_count": 50,
                "features": DEFAULT_OBSERVATION_FEATURES,
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-30, 30],
                    "vy": [-20, 20],
                },
                "grid_size": DEFAULT_GRID_SIZE,
                "grid_step": DEFAULT_GRID_STEP,
                "absolute": False,
                "align_to_vehicle_axes": True,
                "include_ego_vehicle": True,
                "on_road_mode": "area",
                "on_road_soft_mode": True,
                "presence_subsamples": 5,
                "on_road_subsamples": 3,
            },
            "collision_reward": 0,
            "left_lane_constraint": 0,
            "left_lane_reward": 0,
            "high_speed_reward": 0,
            "vehicles_density": vehicle_density,
            "action": {
                "type": "DiscreteAction",
                "longitudinal": True,
                "lateral": True,
                "actions_per_axis": 5,  # 5x5 = 25 discrete actions
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-np.pi / 6, np.pi / 6],
            },
            "lanes_count": 4,
            "policy_frequency": 1,
            "duration": 40,
        }

    @classmethod
    def create(cls, env_name="highway-v0", vehicle_density=1, seed=21, render_mode="rgb_array"):
        env_config = cls.default_config(vehicle_density=vehicle_density)
        env = gym.make(env_name, config=env_config, render_mode=render_mode)
        env.reset(seed=seed)
        return env


def init_env(env_name="highway-v0", vehicle_density=1, seed=21):
    return EnvironmentFactory.create(env_name=env_name, vehicle_density=vehicle_density, seed=seed)

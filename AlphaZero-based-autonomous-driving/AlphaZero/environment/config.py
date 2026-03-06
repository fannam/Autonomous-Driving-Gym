import gymnasium as gym
import highway_env
import numpy as np


class EnvironmentFactory:
    @staticmethod
    def default_config(vehicle_density=1):
        return {
            "observation": {
                "type": "OccupancyGrid",
                "vehicles_count": 50,
                "features": ["presence", "on_road"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-30, 30],
                    "vy": [-20, 20],
                },
                "grid_size": [[-20, 85], [-12.5, 12.5]],
                "grid_step": [5, 5],
                "absolute": False,
            },
            "collision_reward": 0,
            "left_lane_constraint": 0,
            "left_lane_reward": 0,
            "high_speed_reward": 0,
            "vehicles_density": vehicle_density,
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(20, 30, 3),
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

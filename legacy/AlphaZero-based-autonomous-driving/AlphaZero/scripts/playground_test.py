import time

import gymnasium as gym
import highway_env
import numpy as np


def run_action_playground(seed=70, action_list=None):
    if action_list is None:
        # For 5x5 action grid, id=12 is roughly (acc=0, steer=0).
        action_list = [12, 12, 12, 12, 12, 17, 17, 12, 7, 7, 12]

    env_config = {
        "observation": {
            "type": "DetailedOccupancyGrid",
            "vehicles_count": 50,
            "features": ["presence", "on_lane", "on_road"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-30, 30],
                "vy": [-20, 20],
            },
            "grid_size": [[-50, 50], [-12, 12]],
            "grid_step": [1.0, 1.0],
            "absolute": False,
            "align_to_vehicle_axes": True,
            "include_ego_vehicle": True,
            "on_road_mode": "area",
            "presence_subsamples": 5,
            "on_road_soft_mode": True,
            "on_road_subsamples": 3,
        },
        "collision_reward": 0,
        "left_lane_constraint": 0,
        "left_lane_reward": 0,
        "high_speed_reward": 0,
        "lanes_count": 4,
        "vehicles_density": 1.6 + 2 * np.random.rand(),
        "action": {
            "type": "DiscreteAction",
            "longitudinal": True,
            "lateral": True,
            "actions_per_axis": 5,
            "acceleration_range": [-5.0, 5.0],
            "steering_range": [-np.pi / 6, np.pi / 6],
        },
        "duration": 12,
    }

    env = gym.make("highway-fast-v0", config=env_config, render_mode="rgb_array")
    env.reset(seed=seed)

    for vehicle in env.unwrapped.road.vehicles:
        if isinstance(vehicle, highway_env.vehicle.controller.MDPVehicle):
            mdp_vehicle = vehicle
            break
    else:
        mdp_vehicle = env.unwrapped.road.vehicles[0]

    speeds = []
    positions = []
    for action in action_list:
        env.step(action)
        speed = mdp_vehicle.speed
        pos = env.unwrapped.road.vehicles[0].position[0]
        positions.append(pos)
        speeds.append(speed)
        print(f"action: {action}, speed: {speed}")
        print(env.unwrapped.get_available_actions())
        env.render()
        time.sleep(0.3)

    print(positions)
    print((positions[-1] - positions[0]) / (len(action_list) - 1))
    print(f"Terminated = {env.unwrapped._is_terminated()}")
    print(f"Truncated = {env.unwrapped._is_truncated()}")
    print(np.mean(speeds))


def main():
    run_action_playground()


if __name__ == "__main__":
    main()

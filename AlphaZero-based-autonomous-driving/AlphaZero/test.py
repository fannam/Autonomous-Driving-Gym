import highway_env
import gymnasium as gym
import numpy as np
import time  # Thêm thư viện time để sử dụng sleep
import copy

action_list = [1,1,4,0,3,3,2,0,0,2,4,4,4,3,2,4,3,4,3,4,3,0,3,0,3,1,2,4,1,3]
action_list = [2,1,1,1,1,1,1,1,1,1,3,1,1]


env_config = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 50,
        "features": ["presence", "on_road"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-30, 30],
            "vy": [-20, 20]
        },
        "grid_size": [[-20, 85], [-12.5, 12.5]],
        "grid_step": [5, 5],
        "absolute": False
    },
    "collision_reward": 0,
    "left_lane_constraint": 0,
    "left_lane_reward": 0,
    "high_speed_reward": 0,
    "lanes_count": 4,
    "vehicles_density": 1.6+2*np.random.rand(),
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": np.linspace(15, 30, 4)
    },
    "reward_speed_range": np.array([15, 30]),
    "duration": 12
}

env = gym.make("highway-fast-v0", config=env_config, render_mode='rgb_array')
env.reset(seed=70)
for vehicle in env.unwrapped.road.vehicles:
    if isinstance(vehicle, highway_env.vehicle.controller.MDPVehicle):
        mdp_vehicle = vehicle
        break
speeds = []
positions = []
for a in action_list:
    start_time = time.time()  # Lấy thời gian hiện tại
    env.step(a)  # Thực hiện bước hành động
    speed = mdp_vehicle.speed
    pos = env.unwrapped.road.vehicles[0].position[0]
    positions.append(pos)
    speeds.append(speed)
    print(f"action: {a}, speed: {speed}")
    print(env.unwrapped.get_available_actions())
    #print(env.unwrapped.road.vehicles)
    env.render()
    time.sleep(0.3)
print(positions)
print((positions[-1]-positions[0])/(len(action_list)-1))

print(f"Terminated = {env.unwrapped._is_terminated()}")
print(f"Truncated = {env.unwrapped._is_truncated()}")
print(np.mean(speeds))

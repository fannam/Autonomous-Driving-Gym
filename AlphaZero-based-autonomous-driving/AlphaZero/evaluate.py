from CNN_alphazero import AlphaZeroNetwork
from stack_of_planes_9_layers import init_stack_of_grid, get_stack_of_grid
from policy_smoothing import softmax_policy
from trainer import AlphaZeroTrainer
from env_config import init_env
import torch
import numpy as np
import copy

ego_position=(4,2)
grid_size=(21,5)
model_path="alphazero_model (19).pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = AlphaZeroNetwork(input_shape=(21, 5, 9))
number_of_step = []
average_speed = []

def evaluate(network, seed):
    action_list = []
    speed_list = []
    state = init_stack_of_grid(grid_size, ego_position)
    env = init_env(env_name='highway-v0', vehicle_density=1.0, seed=seed)
    trainer = AlphaZeroTrainer(network, env, c_puct=3.5, n_simulations=15, learning_rate=0.001, batch_size=64, epochs=30)
    trainer.load_model(model_path)
    trainer.network.eval()
    ego_position_list = []
    ego_position_list.append(env.unwrapped.road.vehicles[0].position[0])
    while not env.unwrapped._is_terminated() and not env.unwrapped._is_truncated():
        obs = env.unwrapped.observation_type.observe()
        state = get_stack_of_grid(env, state, obs)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        predicted_policy, predicted_value = trainer.network(state_tensor)
        available_actions = env.unwrapped.get_available_actions()
        predicted_policy = {action: prob for action, prob in enumerate(predicted_policy.squeeze().tolist())}
        updated_policy = softmax_policy(predicted_policy, available_actions)
        action = max(updated_policy, key=updated_policy.get)
        env.render()
        #print(f"action chosen: {action}")
        env.step(action)
    #     ego_position_list.append(env.unwrapped.road.vehicles[0].position[0])
    #     action_list.append(action)
    #     speed_list.append(env.unwrapped.road.vehicles[0].speed)
    # number_of_step.append(len(action_list))
    # average_speed.append((ego_position_list[-1]-ego_position_list[0])/len(action_list)-1)
seed = [80, 81, 94]
for seed in range(80, 100):
    evaluate(network=network ,seed=seed)

# for i in range(len(average_speed)):
#     print(f"steps: {number_of_step[i]}, avg speed: {average_speed[i]}")
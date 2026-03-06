from CNN_alphazero import AlphaZeroNetwork
from stack_of_planes import init_stack_of_grid, get_stack_of_grid
from policy_smoothing import softmax_policy
from trainer import AlphaZeroTrainer
from env_config import init_env
import torch

ego_position=(4,2)
grid_size=(21,5)
model_path="alphazero_model (19).pth"
env = init_env(seed=28)

network = AlphaZeroNetwork(input_shape=(21, 5, 9))

trainer = AlphaZeroTrainer(network, env, c_puct=3.5, n_simulations=15, learning_rate=0.001, batch_size=64, epochs=30)
trainer.load_model(model_path)
action_list = []
state = init_stack_of_grid(grid_size, ego_position)

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
    env.step(action)


from trainer import AlphaZeroTrainer
from CNN_alphazero import AlphaZeroNetwork
from env_config import init_env
import pickle

env = init_env(seed=10)

network = AlphaZeroNetwork(input_shape = (6, 21, 5), n_residual_layers=10, n_actions=5)

trainer = AlphaZeroTrainer(network, env, 2.5, 5, 0.001, 32, 10)

trainer.self_play()


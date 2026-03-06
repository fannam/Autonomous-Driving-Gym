import torch

try:
    from core.policy import softmax_policy
    from core.state_stack import get_stack_of_grid_9_layers, init_stack_of_grid_9_layers
    from environment.config import init_env
    from network.alphazero_network import AlphaZeroNetwork
    from training.trainer import AlphaZeroTrainer
except ModuleNotFoundError as exc:
    if exc.name not in {"core", "environment", "network", "training"}:
        raise
    from ..core.policy import softmax_policy
    from ..core.state_stack import get_stack_of_grid_9_layers, init_stack_of_grid_9_layers
    from ..environment.config import init_env
    from ..network.alphazero_network import AlphaZeroNetwork
    from ..training.trainer import AlphaZeroTrainer

EGO_POSITION = (4, 2)
GRID_SIZE = (21, 5)
MODEL_PATH = "alphazero_model (19).pth"


def evaluate_episode(network, seed, model_path=MODEL_PATH):
    state = init_stack_of_grid_9_layers(GRID_SIZE, EGO_POSITION)
    env = init_env(env_name="highway-v0", vehicle_density=1.0, seed=seed)
    trainer = AlphaZeroTrainer(
        network=network,
        env=env,
        c_puct=3.5,
        n_simulations=15,
        learning_rate=0.001,
        batch_size=64,
        epochs=30,
    )
    trainer.load_model(model_path)
    trainer.network.eval()

    while not env.unwrapped._is_terminated() and not env.unwrapped._is_truncated():
        observation = env.unwrapped.observation_type.observe()
        state = get_stack_of_grid_9_layers(env, state, observation)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        predicted_policy, _ = trainer.network(state_tensor)
        available_actions = env.unwrapped.get_available_actions()
        predicted_policy = {action: prob for action, prob in enumerate(predicted_policy.squeeze().tolist())}
        updated_policy = softmax_policy(predicted_policy, available_actions)
        action = max(updated_policy, key=updated_policy.get)
        env.render()
        env.step(action)


def run_batch_evaluation(seed_start=80, seed_end=100):
    network = AlphaZeroNetwork(input_shape=(21, 5, 9))
    for seed in range(seed_start, seed_end):
        evaluate_episode(network=network, seed=seed)


def main():
    run_batch_evaluation()


if __name__ == "__main__":
    main()

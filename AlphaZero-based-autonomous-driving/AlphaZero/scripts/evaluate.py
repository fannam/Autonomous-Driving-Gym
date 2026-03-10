import torch

try:
    from core.policy import softmax_policy
    from core.settings import EVALUATION_CONFIG
    from core.state_stack import init_state_stack, update_state_stack
    from environment.config import init_env
    from network.alphazero_network import AlphaZeroNetwork
    from training.trainer import AlphaZeroTrainer
except ModuleNotFoundError as exc:
    if exc.name not in {"core", "environment", "network", "training"}:
        raise
    from ..core.policy import softmax_policy
    from ..core.settings import EVALUATION_CONFIG
    from ..core.state_stack import init_state_stack, update_state_stack
    from ..environment.config import init_env
    from ..network.alphazero_network import AlphaZeroNetwork
    from ..training.trainer import AlphaZeroTrainer

DEFAULT_CONFIG = EVALUATION_CONFIG


def evaluate_episode(network, seed, model_path=None, config=None, device="auto"):
    if config is None:
        config = DEFAULT_CONFIG
    if model_path is None:
        model_path = config.model_path
    state = init_state_stack(config.stack)
    env = init_env(env_name="highway-v0", vehicle_density=1.0, seed=seed)
    trainer = AlphaZeroTrainer(
        network=network,
        env=env,
        c_puct=config.c_puct,
        n_simulations=config.n_simulations,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        epochs=config.epochs,
        stack_config=config.stack,
        n_actions=config.n_actions,
        device=device,
    )
    trainer.load_model(model_path)
    trainer.network.eval()

    while not env.unwrapped._is_terminated() and not env.unwrapped._is_truncated():
        observation = env.unwrapped.observation_type.observe()
        state = update_state_stack(env, state, observation, stack_config=config.stack)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(trainer.device)
        predicted_policy, _ = trainer.network(state_tensor)
        available_actions = env.unwrapped.get_available_actions()
        predicted_policy = {action: prob for action, prob in enumerate(predicted_policy.squeeze().tolist())}
        updated_policy = softmax_policy(predicted_policy, available_actions)
        action = max(updated_policy, key=updated_policy.get)
        env.render()
        env.step(action)


def run_batch_evaluation(seed_start=80, seed_end=100, config=None, device="auto"):
    if config is None:
        config = DEFAULT_CONFIG
    network = AlphaZeroNetwork(
        input_shape=config.input_shape,
        n_residual_layers=config.n_residual_layers,
        n_actions=config.n_actions,
    )
    for seed in range(seed_start, seed_end):
        evaluate_episode(network=network, seed=seed, config=config, device=device)


def main():
    run_batch_evaluation()


if __name__ == "__main__":
    main()

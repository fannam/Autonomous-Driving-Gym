try:
    from core.settings import INFERENCE_CONFIG
    from environment.config import init_env
    from network.alphazero_network import AlphaZeroNetwork
    from training.trainer import AlphaZeroTrainer
except ModuleNotFoundError as exc:
    if exc.name not in {"core", "environment", "network", "training"}:
        raise
    from ..core.settings import INFERENCE_CONFIG
    from ..environment.config import init_env
    from ..network.alphazero_network import AlphaZeroNetwork
    from ..training.trainer import AlphaZeroTrainer

DEFAULT_CONFIG = INFERENCE_CONFIG


def run_inference(model_path=None, seed=28, config=None, device="auto"):
    if config is None:
        config = DEFAULT_CONFIG
    if model_path is None:
        model_path = config.model_path

    env = init_env(seed=seed)
    network = AlphaZeroNetwork(
        input_shape=config.input_shape,
        n_residual_layers=config.n_residual_layers,
        n_actions=config.n_actions,
    )
    trainer = AlphaZeroTrainer(
        network=network,
        env=env,
        c_puct=config.c_puct,
        n_simulations=config.n_simulations,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        epochs=config.epochs,
        stack_config=config.stack,
        n_actions=config.n_actions,
        device=device,
        temperature=config.temperature,
        temperature_drop_step=config.temperature_drop_step,
        add_root_dirichlet_noise=False,
        root_dirichlet_alpha=config.root_dirichlet_alpha,
        root_exploration_fraction=config.root_exploration_fraction,
    )
    trainer.load_model(model_path)

    while not env.unwrapped._is_terminated() and not env.unwrapped._is_truncated():
        action, _ = trainer.choose_action(
            temperature=config.temperature,
            add_root_dirichlet_noise=False,
            sample_from_policy=False,
        )
        env.render()
        env.step(action)


def main():
    run_inference()


if __name__ == "__main__":
    main()

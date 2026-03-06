try:
    from environment.config import init_env
    from network.alphazero_network import AlphaZeroNetwork
    from training.trainer import AlphaZeroTrainer
except ModuleNotFoundError as exc:
    if exc.name not in {"environment", "network", "training"}:
        raise
    from ..environment.config import init_env
    from ..network.alphazero_network import AlphaZeroNetwork
    from ..training.trainer import AlphaZeroTrainer


def run_self_play(
    env_seed=10,
    self_play_seed=21,
    input_shape=(6, 21, 5),
    n_residual_layers=10,
    n_actions=5,
    c_puct=2.5,
    n_simulations=5,
    learning_rate=0.001,
    batch_size=32,
    epochs=10,
):
    env = init_env(seed=env_seed)
    network = AlphaZeroNetwork(
        input_shape=input_shape,
        n_residual_layers=n_residual_layers,
        n_actions=n_actions,
    )
    trainer = AlphaZeroTrainer(
        network=network,
        env=env,
        c_puct=c_puct,
        n_simulations=n_simulations,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
    )
    trainer.self_play(seed=self_play_seed)
    return trainer


def main():
    run_self_play()


if __name__ == "__main__":
    main()

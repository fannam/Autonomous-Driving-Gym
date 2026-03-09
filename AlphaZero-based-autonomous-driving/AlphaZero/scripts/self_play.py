try:
    from core.settings import SELF_PLAY_CONFIG
    from environment.config import init_env
    from network.alphazero_network import AlphaZeroNetwork
    from training.trainer import AlphaZeroTrainer
except ModuleNotFoundError as exc:
    if exc.name not in {"core", "environment", "network", "training"}:
        raise
    from ..core.settings import SELF_PLAY_CONFIG
    from ..environment.config import init_env
    from ..network.alphazero_network import AlphaZeroNetwork
    from ..training.trainer import AlphaZeroTrainer


def run_self_play(
    env_seed=10,
    self_play_seed=21,
    input_shape=None,
    n_residual_layers=None,
    n_actions=None,
    c_puct=None,
    n_simulations=None,
    learning_rate=None,
    batch_size=None,
    epochs=None,
    stack_config=None,
):
    config = SELF_PLAY_CONFIG
    if input_shape is None:
        input_shape = config.input_shape
    if n_residual_layers is None:
        n_residual_layers = config.n_residual_layers
    if n_actions is None:
        n_actions = config.n_actions
    if c_puct is None:
        c_puct = config.c_puct
    if n_simulations is None:
        n_simulations = config.n_simulations
    if learning_rate is None:
        learning_rate = config.learning_rate
    if batch_size is None:
        batch_size = config.batch_size
    if epochs is None:
        epochs = config.epochs
    if stack_config is None:
        stack_config = config.stack

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
        stack_config=stack_config,
        n_actions=n_actions,
    )
    trainer.self_play(seed=self_play_seed)
    return trainer


def main():
    run_self_play()


if __name__ == "__main__":
    main()

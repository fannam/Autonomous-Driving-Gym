import argparse

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
    device="auto",
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
        device=device,
    )
    trainer.self_play(seed=self_play_seed)
    return trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-process AlphaZero self-play.")
    parser.add_argument("--env-seed", type=int, default=10)
    parser.add_argument("--self-play-seed", type=int, default=21)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto|cpu|cuda|cuda:0",
    )
    parser.add_argument("--n-residual-layers", type=int, default=None)
    parser.add_argument("--c-puct", type=float, default=None)
    parser.add_argument("--n-simulations", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_self_play(
        env_seed=args.env_seed,
        self_play_seed=args.self_play_seed,
        n_residual_layers=args.n_residual_layers,
        c_puct=args.c_puct,
        n_simulations=args.n_simulations,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()

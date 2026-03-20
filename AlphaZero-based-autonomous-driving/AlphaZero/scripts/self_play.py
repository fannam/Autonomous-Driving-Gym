import argparse
import sys
from pathlib import Path

try:
    from core.settings import SELF_PLAY_CONFIG
    from environment.config import init_env
    from network.alphazero_network import AlphaZeroNetwork
    from training.trainer import AlphaZeroTrainer
except ModuleNotFoundError as exc:
    if exc.name not in {"core", "environment", "network", "training", "AlphaZero"}:
        raise
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from core.settings import SELF_PLAY_CONFIG
    from environment.config import init_env
    from network.alphazero_network import AlphaZeroNetwork
    from training.trainer import AlphaZeroTrainer


def run_self_play(
    env_seed=10,
    self_play_seed=21,
    input_shape=None,
    n_residual_layers=None,
    n_actions=None,
    c_puct=None,
    n_simulations=None,
    stack_config=None,
    device="auto",
    max_root_visits=None,
    temperature=None,
    temperature_drop_step=None,
    root_dirichlet_alpha=None,
    root_exploration_fraction=None,
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
    if stack_config is None:
        stack_config = config.stack
    if temperature is None:
        temperature = config.temperature
    if temperature_drop_step is None:
        temperature_drop_step = config.temperature_drop_step
    if root_dirichlet_alpha is None:
        root_dirichlet_alpha = config.root_dirichlet_alpha
    if root_exploration_fraction is None:
        root_exploration_fraction = config.root_exploration_fraction

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
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        stack_config=stack_config,
        n_actions=n_actions,
        device=device,
        max_root_visits=max_root_visits,
        temperature=temperature,
        temperature_drop_step=temperature_drop_step,
        add_root_dirichlet_noise=True,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_exploration_fraction=root_exploration_fraction,
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
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--temperature-drop-step", type=int, default=None)
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    parser.add_argument("--root-exploration-fraction", type=float, default=None)
    parser.add_argument(
        "--mcts-max-root-visits",
        type=int,
        default=None,
        help="Stop rollouts for current decision once root visit count reaches this cap.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_self_play(
        env_seed=args.env_seed,
        self_play_seed=args.self_play_seed,
        n_residual_layers=args.n_residual_layers,
        c_puct=args.c_puct,
        n_simulations=args.n_simulations,
        device=args.device,
        max_root_visits=args.mcts_max_root_visits,
        temperature=args.temperature,
        temperature_drop_step=args.temperature_drop_step,
        root_dirichlet_alpha=args.dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
    )


if __name__ == "__main__":
    main()

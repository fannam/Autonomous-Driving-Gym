import argparse
import sys
from pathlib import Path

try:
    from AlphaZeroAdversarial.environment.config import init_env
    from AlphaZeroAdversarial.network.alphazero_network import AlphaZeroNetwork
    from AlphaZeroAdversarial.core.settings import EVALUATION_CONFIG
    from AlphaZeroAdversarial.training.trainer import AdversarialAlphaZeroTrainer
except ModuleNotFoundError:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))
    from AlphaZeroAdversarial.environment.config import init_env
    from AlphaZeroAdversarial.network.alphazero_network import AlphaZeroNetwork
    from AlphaZeroAdversarial.core.settings import EVALUATION_CONFIG
    from AlphaZeroAdversarial.training.trainer import AdversarialAlphaZeroTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved adversarial AlphaZero model.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--env-seed", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = EVALUATION_CONFIG
    env = init_env(seed=args.env_seed, stage="evaluation")
    network = AlphaZeroNetwork(
        input_shape=config.input_shape,
        n_residual_layers=config.n_residual_layers,
        n_actions=config.n_actions,
        n_action_axis_0=config.n_action_axis_0,
        n_action_axis_1=config.n_action_axis_1,
        channels=config.network_channels,
        dropout_p=config.network_dropout_p,
    )
    trainer = AdversarialAlphaZeroTrainer(
        network=network,
        config=config,
        env=env,
        device=args.device,
        verbose=not args.quiet,
        add_root_dirichlet_noise=False,
    )
    trainer.load_model(args.model_path)

    summaries = []
    for episode_offset in range(args.episodes):
        summaries.append(
            trainer.run_episode(
                seed=args.seed_start + episode_offset,
                episode_index=config.warmup_episodes + episode_offset,
                max_steps=args.max_steps,
                store_in_replay=False,
                add_root_dirichlet_noise=False,
                sample_actions=False,
            )
        )
    print(summaries)


if __name__ == "__main__":
    main()

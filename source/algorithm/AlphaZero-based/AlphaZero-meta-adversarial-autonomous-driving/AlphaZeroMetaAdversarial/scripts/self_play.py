import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

try:
    from AlphaZeroMetaAdversarial.environment.config import init_env
    from AlphaZeroMetaAdversarial.network.alphazero_network import AlphaZeroNetwork
    from AlphaZeroMetaAdversarial.core.settings import SELF_PLAY_CONFIG
    from AlphaZeroMetaAdversarial.training.trainer import AdversarialAlphaZeroTrainer
except ModuleNotFoundError:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))
    from AlphaZeroMetaAdversarial.environment.config import init_env
    from AlphaZeroMetaAdversarial.network.alphazero_network import AlphaZeroNetwork
    from AlphaZeroMetaAdversarial.core.settings import SELF_PLAY_CONFIG
    from AlphaZeroMetaAdversarial.training.trainer import AdversarialAlphaZeroTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run one meta-adversarial AlphaZero self-play episode.")
    parser.add_argument("--env-seed", type=int, default=10)
    parser.add_argument("--episode-seed", type=int, default=21)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--discount-gamma", type=float, default=None)
    parser.add_argument("--network-seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = SELF_PLAY_CONFIG
    if args.discount_gamma is not None:
        config = replace(config, discount_gamma=float(args.discount_gamma))
    env = init_env(seed=args.env_seed, stage="self_play")
    torch.manual_seed(int(args.network_seed))
    network = AlphaZeroNetwork(
        input_shape=config.input_shape,
        n_residual_layers=config.n_residual_layers,
        n_actions=config.n_actions,
        channels=config.network_channels,
        dropout_p=config.network_dropout_p,
        target_vector_dim=config.target_vector_dim,
        target_hidden_dim=config.target_hidden_dim,
    )
    trainer = AdversarialAlphaZeroTrainer(
        network=network,
        config=config,
        env=env,
        device=args.device,
        verbose=not args.quiet,
        add_root_dirichlet_noise=True,
    )
    if args.model_path:
        trainer.load_model(args.model_path)
    summary = trainer.run_episode(
        seed=args.episode_seed,
        env_seed=args.env_seed,
        episode_index=args.episode_index,
        max_steps=args.max_steps,
        store_in_replay=False,
        add_root_dirichlet_noise=True,
        sample_actions=True,
    )
    summary["network_seed"] = int(args.network_seed)
    summary["model_path"] = args.model_path
    print(summary)


if __name__ == "__main__":
    main()

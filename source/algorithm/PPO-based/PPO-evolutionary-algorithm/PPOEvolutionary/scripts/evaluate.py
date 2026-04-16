from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PPOEvolutionary.training.trainer import PPOEvolutionaryTrainer
except ModuleNotFoundError:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))
    from PPOEvolutionary.training.trainer import PPOEvolutionaryTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO-evolutionary checkpoint.")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--policy-source", type=str, default="best", choices=("best", "ppo"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = PPOEvolutionaryTrainer(
        config_path=args.config_path,
        device=args.device,
        verbose=not args.quiet,
    )
    summary = trainer.evaluate(
        checkpoint_path=args.checkpoint_path,
        policy_source=args.policy_source,
        episodes=args.episodes,
        seed_start=args.seed_start,
        render_mode=args.render_mode,
    )
    print(summary)


if __name__ == "__main__":
    main()

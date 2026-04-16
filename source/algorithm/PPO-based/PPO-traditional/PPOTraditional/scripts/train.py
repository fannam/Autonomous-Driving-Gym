from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PPOTraditional.training.trainer import PPOTraditionalTrainer
except ModuleNotFoundError:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))
    from PPOTraditional.training.trainer import PPOTraditionalTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PPO-traditional highway agent.")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--steps-per-env", type=int, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed-start", type=int, default=21)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = PPOTraditionalTrainer(
        config_path=args.config_path,
        device=args.device,
        verbose=not args.quiet,
    )
    summaries = trainer.fit(
        updates=args.updates,
        n_envs=args.n_envs,
        steps_per_env=args.steps_per_env,
        seed_start=args.seed_start,
        save_path=args.save_path,
    )
    checkpoint_path = trainer.save_checkpoint(args.save_path)
    print(
        {
            "checkpoint_path": str(checkpoint_path),
            "updates": len(summaries),
        }
    )


if __name__ == "__main__":
    main()

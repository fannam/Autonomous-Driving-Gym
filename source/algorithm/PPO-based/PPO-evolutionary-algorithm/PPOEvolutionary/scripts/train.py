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
    parser = argparse.ArgumentParser(description="Train the PPO-evolutionary highway agent.")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed-start", type=int, default=21)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = PPOEvolutionaryTrainer(
        config_path=args.config_path,
        device=args.device,
        verbose=not args.quiet,
    )
    summaries = trainer.fit(
        generations=args.generations,
        population_size=args.population_size,
        workers=args.workers,
        seed_start=args.seed_start,
        save_path=args.save_path,
    )
    checkpoint_path = trainer.save_checkpoint(args.save_path)
    print(
        {
            "checkpoint_path": str(checkpoint_path),
            "generations": len(summaries),
        }
    )


if __name__ == "__main__":
    main()

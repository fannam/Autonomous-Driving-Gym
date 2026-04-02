import argparse
import json
import sys
from pathlib import Path

import torch

try:
    from core.settings import ACTIVE_SCENARIO, CONFIG_PATH, SELF_PLAY_CONFIG
    from network.alphazero_network import AlphaZeroNetwork
    from training.trainer import AlphaZeroTrainer
except ModuleNotFoundError as exc:
    if exc.name not in {"core", "network", "training", "AlphaZero"}:
        raise
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from core.settings import ACTIVE_SCENARIO, CONFIG_PATH, SELF_PLAY_CONFIG
    from network.alphazero_network import AlphaZeroNetwork
    from training.trainer import AlphaZeroTrainer


def _resolve_input_shape(state_tensor: torch.Tensor) -> tuple[int, int, int]:
    if state_tensor.ndim != 4:
        raise ValueError(
            f"Expected states with shape (batch, channels, width, height), got {tuple(state_tensor.shape)}."
        )
    _, channels, width, height = state_tensor.shape
    return (int(width), int(height), int(channels))


def _load_episode_file(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location=torch.device("cpu"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a serialized dictionary payload.")

    missing_keys = {"states", "policies", "values"} - set(payload.keys())
    if missing_keys:
        raise ValueError(f"{path} is missing required keys: {sorted(missing_keys)}")

    states = torch.as_tensor(payload["states"], dtype=torch.float32)
    policies = torch.as_tensor(payload["policies"], dtype=torch.float32)
    values = torch.as_tensor(payload["values"], dtype=torch.float32)
    if values.ndim == 1:
        values = values.unsqueeze(1)
    return states, policies, values


def load_self_play_dataset(
    input_dir,
    pattern="*.pt",
    recursive=False,
    limit_files=None,
):
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    episode_paths = sorted(
        input_path.rglob(pattern) if recursive else input_path.glob(pattern)
    )
    if limit_files is not None:
        episode_paths = episode_paths[: int(limit_files)]
    if not episode_paths:
        raise FileNotFoundError(
            f"No self-play episode files matched pattern {pattern!r} in {input_path}"
        )

    states_list = []
    policies_list = []
    values_list = []

    for episode_path in episode_paths:
        states, policies, values = _load_episode_file(episode_path)
        states_list.append(states)
        policies_list.append(policies)
        values_list.append(values)

    states = torch.cat(states_list, dim=0)
    policies = torch.cat(policies_list, dim=0)
    values = torch.cat(values_list, dim=0)
    return episode_paths, states, policies, values


def parse_args() -> argparse.Namespace:
    config = SELF_PLAY_CONFIG
    parser = argparse.ArgumentParser(
        description="Train an AlphaZero network from saved self-play episode tensors."
    )
    parser.add_argument(
        "--input-dir",
        default="legacy/AlphaZero-based-autonomous-driving/outputs/self_play_parallel",
        help="Directory containing per-episode .pt files produced by the parallel self-play script.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pt",
        help="Glob used to select episode files inside --input-dir.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for episode files under --input-dir.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional cap on the number of episode files loaded.",
    )
    parser.add_argument(
        "--model-in",
        default=None,
        help="Optional checkpoint to initialize from before training.",
    )
    parser.add_argument(
        "--model-out",
        default="legacy/AlphaZero-based-autonomous-driving/outputs/alphazero_from_self_play.pth",
        help="Checkpoint path to write after training.",
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Optional JSON path for epoch metrics. Defaults to <model-out>.metrics.json.",
    )
    parser.add_argument("--n-residual-layers", type=int, default=config.n_residual_layers)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--learning-rate", type=float, default=config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=config.weight_decay)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto|cpu|cuda|cuda:0",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable dataset shuffling during training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    episode_paths, states, policies, values = load_self_play_dataset(
        input_dir=args.input_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        limit_files=args.limit_files,
    )

    input_shape = _resolve_input_shape(states)
    n_actions = int(policies.shape[1])
    network = AlphaZeroNetwork(
        input_shape=input_shape,
        n_residual_layers=args.n_residual_layers,
        n_actions=n_actions,
    )

    if args.model_in is not None:
        model_in = Path(args.model_in).expanduser().resolve()
        network.load_state_dict(torch.load(model_in, map_location=torch.device("cpu")))
    else:
        model_in = None

    trainer = AlphaZeroTrainer(
        network=network,
        env=None,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_actions=n_actions,
        verbose=True,
        device=args.device,
    )

    metrics = trainer.train_from_tensors(
        states,
        policies,
        values,
        shuffle=not args.no_shuffle,
    )

    model_out = Path(args.model_out).expanduser().resolve()
    trainer.save_model(model_out)

    metrics_out = (
        Path(args.metrics_out).expanduser().resolve()
        if args.metrics_out is not None
        else model_out.with_suffix(model_out.suffix + ".metrics.json")
    )
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"loaded_files={len(episode_paths)}")
    print(f"loaded_samples={int(states.shape[0])}")
    print(f"config_path={CONFIG_PATH}")
    print(f"active_scenario={ACTIVE_SCENARIO}")
    print(f"input_shape={input_shape}")
    print(f"n_actions={n_actions}")
    if model_in is not None:
        print(f"model_in={model_in}")
    print(f"model_out={model_out}")
    print(f"metrics_out={metrics_out}")


if __name__ == "__main__":
    main()

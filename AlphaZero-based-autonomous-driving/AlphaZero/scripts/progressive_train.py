import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
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


DEFAULT_INPUT_ROOT = Path(
    "AlphaZero-based-autonomous-driving/outputs/progressive_self_play"
)
DEFAULT_OUTPUT_ROOT = Path(
    "AlphaZero-based-autonomous-driving/outputs/progressive_training"
)
DEFAULT_MANIFEST_NAME = "manifest.json"


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def _extract_iteration_from_name(path: Path) -> int | None:
    match = re.search(r"iter[_-]?(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_batches(input_root: Path, manifest_name: str) -> list[tuple[Path, dict | None]]:
    discovered = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        manifest_path = child / manifest_name
        manifest = _read_manifest(manifest_path) if manifest_path.exists() else None
        discovered.append((child, manifest))
    return discovered


def _sort_batch_entries(entries: list[tuple[Path, dict | None]]) -> list[tuple[Path, dict | None]]:
    def sort_key(item: tuple[Path, dict | None]) -> tuple[int, str]:
        batch_dir, manifest = item
        if manifest is not None:
            iteration = manifest.get("iteration")
            if isinstance(iteration, int):
                return (iteration, batch_dir.name)
        inferred_iteration = _extract_iteration_from_name(batch_dir)
        return (-1 if inferred_iteration is None else inferred_iteration, batch_dir.name)

    return sorted(entries, key=sort_key)


def _resolve_selected_batches(args: argparse.Namespace) -> list[tuple[Path, dict | None]]:
    if args.batch_dir:
        selected = []
        for raw_path in args.batch_dir:
            batch_dir = Path(raw_path).expanduser().resolve()
            manifest_path = batch_dir / args.manifest_name
            manifest = _read_manifest(manifest_path) if manifest_path.exists() else None
            selected.append((batch_dir, manifest))
        return _sort_batch_entries(selected)

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    discovered = _sort_batch_entries(_discover_batches(input_root, args.manifest_name))
    if not discovered:
        raise FileNotFoundError(
            f"No batch directories were found under {input_root}"
        )

    replay_last_n = int(args.replay_last_n)
    if replay_last_n <= 0:
        raise ValueError("--replay-last-n must be a positive integer.")
    return discovered[-replay_last_n:]


def _resolve_episode_paths(batch_dir: Path, manifest: dict | None) -> list[Path]:
    if manifest is not None:
        episode_entries = manifest.get("episodes")
        if isinstance(episode_entries, list) and episode_entries:
            return [batch_dir / entry["path"] for entry in episode_entries]

    episodes_dir = batch_dir / "episodes"
    if episodes_dir.exists():
        return sorted(episodes_dir.glob("*.pt"))
    return sorted(batch_dir.glob("*.pt"))


def _load_batches(
    selected_batches: list[tuple[Path, dict | None]],
) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor]:
    states_list = []
    policies_list = []
    values_list = []
    batch_summaries = []

    for batch_dir, manifest in selected_batches:
        episode_paths = _resolve_episode_paths(batch_dir, manifest)
        if not episode_paths:
            raise FileNotFoundError(f"No episode files found in batch directory: {batch_dir}")

        batch_states = []
        batch_policies = []
        batch_values = []

        for episode_path in episode_paths:
            states, policies, values = _load_episode_file(episode_path)
            batch_states.append(states)
            batch_policies.append(policies)
            batch_values.append(values)

        batch_state_tensor = torch.cat(batch_states, dim=0)
        batch_policy_tensor = torch.cat(batch_policies, dim=0)
        batch_value_tensor = torch.cat(batch_values, dim=0)

        states_list.append(batch_state_tensor)
        policies_list.append(batch_policy_tensor)
        values_list.append(batch_value_tensor)

        sample_count = int(batch_state_tensor.shape[0])
        batch_summaries.append(
            {
                "batch_dir": str(batch_dir),
                "manifest_iteration": manifest.get("iteration") if manifest else None,
                "episode_file_count": len(episode_paths),
                "sample_count": sample_count,
                "source_model": manifest.get("source_model") if manifest else None,
                "active_scenario": manifest.get("active_scenario") if manifest else None,
                "config_path": manifest.get("config_path") if manifest else None,
            }
        )

    states = torch.cat(states_list, dim=0)
    policies = torch.cat(policies_list, dim=0)
    values = torch.cat(values_list, dim=0)
    return batch_summaries, states, policies, values


def _validate_batch_scenarios(
    selected_batches: list[tuple[Path, dict | None]],
) -> None:
    manifest_scenarios = {
        str(manifest.get("active_scenario"))
        for _, manifest in selected_batches
        if isinstance(manifest, dict) and manifest.get("active_scenario")
    }
    if len(manifest_scenarios) > 1:
        raise ValueError(
            "Selected self-play batches contain multiple scenarios: "
            f"{sorted(manifest_scenarios)}. Train each scenario separately."
        )
    if manifest_scenarios:
        batch_scenario = next(iter(manifest_scenarios))
        if batch_scenario != ACTIVE_SCENARIO:
            raise ValueError(
                "Selected self-play batches were generated for scenario "
                f"{batch_scenario!r}, but the active training scenario is "
                f"{ACTIVE_SCENARIO!r}. Select the matching file under configs/ or set "
                "ALPHAZERO_SCENARIO to match the data you are training on."
            )


def parse_args() -> argparse.Namespace:
    config = SELF_PLAY_CONFIG
    parser = argparse.ArgumentParser(
        description=(
            "Train one progressive AlphaZero iteration on a GPU-oriented machine from "
            "one or more self-play batches produced elsewhere."
        )
    )
    parser.add_argument(
        "--iteration",
        type=int,
        required=True,
        help="Iteration id of the model produced by this training run.",
    )
    parser.add_argument(
        "--model-in",
        required=True,
        help="Checkpoint used to initialize training.",
    )
    parser.add_argument(
        "--batch-dir",
        action="append",
        default=[],
        help="Explicit batch directory to load. Can be repeated.",
    )
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_INPUT_ROOT),
        help="Root directory containing batch subdirectories when --batch-dir is not used.",
    )
    parser.add_argument(
        "--replay-last-n",
        type=int,
        default=1,
        help="Use the latest N discovered batches under --input-root.",
    )
    parser.add_argument(
        "--manifest-name",
        default=DEFAULT_MANIFEST_NAME,
        help="Manifest filename expected inside each batch directory.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory that receives trained checkpoints and training manifests.",
    )
    parser.add_argument(
        "--model-out",
        default=None,
        help="Optional explicit path for the trained checkpoint.",
    )
    parser.add_argument(
        "--train-manifest-out",
        default=None,
        help="Optional explicit path for the JSON training manifest.",
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


def _default_model_out(output_root: Path, iteration: int) -> Path:
    return output_root / f"model_iter_{iteration:04d}.pth"


def _default_manifest_out(model_out: Path) -> Path:
    return model_out.with_suffix(model_out.suffix + ".manifest.json")


def main() -> int:
    args = parse_args()
    model_in = Path(args.model_in).expanduser().resolve()
    if not model_in.exists():
        raise FileNotFoundError(f"Input checkpoint does not exist: {model_in}")

    selected_batches = _resolve_selected_batches(args)
    _validate_batch_scenarios(selected_batches)
    batch_summaries, states, policies, values = _load_batches(selected_batches)

    input_shape = _resolve_input_shape(states)
    n_actions = int(policies.shape[1])
    network = AlphaZeroNetwork(
        input_shape=input_shape,
        n_residual_layers=args.n_residual_layers,
        n_actions=n_actions,
    )
    network.load_state_dict(torch.load(model_in, map_location=torch.device("cpu")))

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

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_out = (
        Path(args.model_out).expanduser().resolve()
        if args.model_out is not None
        else _default_model_out(output_root, args.iteration)
    )
    trainer.save_model(model_out)

    train_manifest_out = (
        Path(args.train_manifest_out).expanduser().resolve()
        if args.train_manifest_out is not None
        else _default_manifest_out(model_out)
    )
    train_manifest_out.parent.mkdir(parents=True, exist_ok=True)

    training_manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now_iso(),
        "config_path": str(CONFIG_PATH),
        "active_scenario": ACTIVE_SCENARIO,
        "training_iteration": int(args.iteration),
        "model_in": {
            "path": str(model_in),
            "filename": model_in.name,
            "sha256": _sha256_file(model_in),
        },
        "model_out": {
            "path": str(model_out),
            "filename": model_out.name,
            "sha256": _sha256_file(model_out),
        },
        "selected_batches": batch_summaries,
        "summary": {
            "batch_count": len(batch_summaries),
            "sample_count": int(states.shape[0]),
            "input_shape": list(input_shape),
            "n_actions": n_actions,
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "device": str(args.device),
        },
        "metrics": metrics,
    }
    train_manifest_out.write_text(
        json.dumps(training_manifest, indent=2),
        encoding="utf-8",
    )

    print(f"loaded_batches={len(batch_summaries)}")
    print(f"loaded_samples={int(states.shape[0])}")
    print(f"config_path={CONFIG_PATH}")
    print(f"active_scenario={ACTIVE_SCENARIO}")
    print(f"input_shape={input_shape}")
    print(f"n_actions={n_actions}")
    print(f"model_in={model_in}")
    print(f"model_out={model_out}")
    print(f"train_manifest={train_manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

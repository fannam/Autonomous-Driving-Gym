import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

try:
    from core.settings import SELF_PLAY_CONFIG
    from network.alphazero_network import AlphaZeroNetwork
except ModuleNotFoundError as exc:
    if exc.name not in {"core", "network", "AlphaZero"}:
        raise
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from core.settings import SELF_PLAY_CONFIG
    from network.alphazero_network import AlphaZeroNetwork


SELF_PLAY_SCRIPT = Path(__file__).resolve().with_name("self_play_parallel_racetrack.py")
DEFAULT_OUTPUT_ROOT = Path(
    "AlphaZero-based-autonomous-driving/outputs/progressive_self_play"
)
DEFAULT_BOOTSTRAP_MODEL_ROOT = Path(
    "AlphaZero-based-autonomous-driving/outputs/progressive_training"
)
DEFAULT_BOOTSTRAP_MODEL_NAME = "model_iter_0000.pth"
DEFAULT_MANIFEST_NAME = "manifest.json"
LOCKED_SELF_PLAY_OPTIONS = ("--model-path", "--output-dir")


def log(message: str) -> None:
    print(f"[progressive-self-play] {message}", flush=True)


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _format_command(parts: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in parts)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _has_option(args: list[str], option_name: str) -> bool:
    return any(arg == option_name or arg.startswith(f"{option_name}=") for arg in args)


def _get_option_value(args: list[str], option_name: str) -> str | None:
    for index, arg in enumerate(args):
        if arg == option_name:
            return args[index + 1] if index + 1 < len(args) else None
        if arg.startswith(f"{option_name}="):
            return arg.split("=", 1)[1]
    return None


def _upsert_option(args: list[str], option_name: str, value: str) -> list[str]:
    updated_args = []
    skip_next = False
    replaced = False

    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == option_name:
            updated_args.extend([option_name, value])
            skip_next = True
            replaced = True
            continue
        if arg.startswith(f"{option_name}="):
            updated_args.extend([option_name, value])
            replaced = True
            continue
        updated_args.append(arg)

    if not replaced:
        updated_args.extend([option_name, value])
    return updated_args


def _load_episode_payload(path: Path) -> dict:
    payload = torch.load(path, map_location=torch.device("cpu"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a serialized dictionary payload.")
    return payload


def _summarize_episode_file(path: Path, batch_dir: Path) -> dict:
    payload = _load_episode_payload(path)
    states = torch.as_tensor(payload["states"], dtype=torch.float32)
    policies = torch.as_tensor(payload["policies"], dtype=torch.float32)
    values = torch.as_tensor(payload["values"], dtype=torch.float32)
    sample_count = int(states.shape[0])
    steps = int(len(payload.get("actions", [])))
    if steps == 0:
        steps = sample_count

    return {
        "path": str(path.relative_to(batch_dir)),
        "worker_id": int(payload.get("worker_id", -1)),
        "episode_index": int(payload.get("episode_index", -1)),
        "self_play_seed": int(payload.get("self_play_seed", -1)),
        "sample_count": sample_count,
        "steps": steps,
        "state_shape": list(states.shape),
        "policy_shape": list(policies.shape),
        "value_shape": list(values.shape),
    }


def _build_batch_dir(output_root: Path, iteration: int, batch_name: str | None) -> Path:
    if batch_name:
        return output_root / batch_name
    return output_root / f"iter_{iteration:04d}"


def _resolve_bootstrap_model_path(raw_path: str | None) -> Path:
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return (DEFAULT_BOOTSTRAP_MODEL_ROOT / DEFAULT_BOOTSTRAP_MODEL_NAME).resolve()


def _get_int_option(args: list[str], option_name: str, default: int) -> int:
    raw_value = _get_option_value(args, option_name)
    return default if raw_value is None else int(raw_value)


def _resolve_source_model(
    args: argparse.Namespace,
    passthrough_args: list[str],
) -> tuple[Path, bool]:
    if args.source_model:
        source_model = Path(args.source_model).expanduser().resolve()
        if not source_model.exists():
            raise FileNotFoundError(f"Source model does not exist: {source_model}")
        return source_model, False

    if int(args.iteration) != 1:
        raise ValueError(
            "--source-model is required for iterations > 1. "
            "Only iteration 1 can auto-bootstrap a fresh checkpoint."
        )

    bootstrap_model = _resolve_bootstrap_model_path(args.bootstrap_model_path)
    if bootstrap_model.exists():
        log(f"Reusing bootstrap checkpoint: {bootstrap_model}")
        return bootstrap_model, True

    n_residual_layers = _get_int_option(
        passthrough_args,
        "--n-residual-layers",
        int(SELF_PLAY_CONFIG.n_residual_layers),
    )
    network_seed = _get_int_option(passthrough_args, "--network-seed", 42)
    bootstrap_model.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(network_seed)
    network = AlphaZeroNetwork(
        input_shape=SELF_PLAY_CONFIG.input_shape,
        n_residual_layers=n_residual_layers,
        n_actions=SELF_PLAY_CONFIG.n_actions,
    )
    torch.save(network.state_dict(), bootstrap_model)
    log(
        "Created bootstrap checkpoint: "
        f"{bootstrap_model} "
        f"(network_seed={network_seed}, n_residual_layers={n_residual_layers})"
    )
    return bootstrap_model, True


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run a progressive self-play batch on a CPU-oriented worker machine and "
            "write a manifest that can be transferred to a GPU training machine."
        )
    )
    parser.add_argument(
        "--iteration",
        type=int,
        required=True,
        help="Iteration id assigned to the self-play batch being generated.",
    )
    parser.add_argument(
        "--source-model",
        default=None,
        help=(
            "Checkpoint used to generate this self-play batch. "
            "If omitted for iteration 1, a fresh bootstrap checkpoint is created/reused."
        ),
    )
    parser.add_argument(
        "--bootstrap-model-path",
        default=None,
        help=(
            "Where to create/reuse the bootstrap checkpoint when --source-model is omitted. "
            f"Defaults to {DEFAULT_BOOTSTRAP_MODEL_ROOT / DEFAULT_BOOTSTRAP_MODEL_NAME}."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory that stores per-iteration self-play batches.",
    )
    parser.add_argument(
        "--batch-name",
        default=None,
        help="Optional directory name for the batch. Defaults to iter_<iteration>.",
    )
    parser.add_argument(
        "--manifest-name",
        default=DEFAULT_MANIFEST_NAME,
        help="Manifest filename written inside the batch directory.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not launch self-play again; only rebuild the manifest from an existing batch directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing batch directory before regenerating it.",
    )
    parser.add_argument(
        "--allow-non-cpu-device",
        action="store_true",
        help="By default this wrapper enforces --device cpu.",
    )
    args, passthrough_args = parser.parse_known_args()
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]
    return args, passthrough_args


def _prepare_passthrough_args(
    passthrough_args: list[str],
    *,
    allow_non_cpu_device: bool,
) -> list[str]:
    for locked_option in LOCKED_SELF_PLAY_OPTIONS:
        if _has_option(passthrough_args, locked_option):
            raise ValueError(
                f"{locked_option} is managed by progressive_self_play.py and must not be passed through."
            )

    effective_args = list(passthrough_args)
    existing_device = _get_option_value(effective_args, "--device")
    if existing_device is None:
        effective_args = _upsert_option(effective_args, "--device", "cpu")
    elif not allow_non_cpu_device and str(existing_device).lower() != "cpu":
        raise ValueError(
            "This wrapper is intended for CPU self-play. "
            "Pass --allow-non-cpu-device if you really want to override --device."
        )

    return effective_args


def _discover_episode_files(episodes_dir: Path) -> list[Path]:
    return sorted(episodes_dir.glob("*.pt"))


def _build_manifest(
    *,
    args: argparse.Namespace,
    batch_dir: Path,
    episodes_dir: Path,
    source_model: Path,
    source_model_sha256: str,
    self_play_command: list[str] | None,
    passthrough_args: list[str],
    episode_files: list[Path],
) -> dict:
    episode_summaries = [
        _summarize_episode_file(path, batch_dir=batch_dir) for path in episode_files
    ]
    total_samples = int(sum(item["sample_count"] for item in episode_summaries))
    total_steps = int(sum(item["steps"] for item in episode_summaries))
    episode_count = int(len(episode_summaries))
    avg_steps_per_episode = 0.0 if episode_count == 0 else total_steps / episode_count

    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now_iso(),
        "hostname": socket.gethostname(),
        "iteration": int(args.iteration),
        "batch_name": batch_dir.name,
        "batch_dir": str(batch_dir),
        "episodes_dir": str(episodes_dir),
        "source_model": {
            "path": str(source_model),
            "filename": source_model.name,
            "sha256": source_model_sha256,
        },
        "self_play_command": self_play_command,
        "self_play_passthrough_args": passthrough_args,
        "summary": {
            "episode_count": episode_count,
            "sample_count": total_samples,
            "total_steps": total_steps,
            "avg_steps_per_episode": avg_steps_per_episode,
        },
        "episodes": episode_summaries,
    }
    return manifest


def main() -> int:
    args, passthrough_args = parse_args()
    source_model, bootstrap_mode = _resolve_source_model(args, passthrough_args)

    output_root = Path(args.output_root).expanduser().resolve()
    batch_dir = _build_batch_dir(output_root, args.iteration, args.batch_name)
    episodes_dir = batch_dir / "episodes"
    manifest_path = batch_dir / args.manifest_name
    source_model_sha256 = _sha256_file(source_model)

    log(
        f"iteration={int(args.iteration)} "
        f"batch_name={batch_dir.name} "
        f"skip_run={bool(args.skip_run)} "
        f"overwrite={bool(args.overwrite)}"
    )
    log(f"source_model={source_model}")
    log(f"source_model_sha256={source_model_sha256}")
    if bootstrap_mode:
        log("source_model_mode=bootstrap")
    log(f"output_root={output_root}")
    log(f"batch_dir={batch_dir}")
    log(f"episodes_dir={episodes_dir}")

    if batch_dir.exists() and not args.skip_run:
        if not args.overwrite:
            raise FileExistsError(
                f"Batch directory already exists: {batch_dir}. "
                "Pass --overwrite to delete and regenerate it."
            )
        log(f"Removing existing batch directory: {batch_dir}")
        shutil.rmtree(batch_dir)

    batch_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    effective_passthrough_args = _prepare_passthrough_args(
        passthrough_args,
        allow_non_cpu_device=bool(args.allow_non_cpu_device),
    )
    resolved_device = _get_option_value(effective_passthrough_args, "--device") or "cpu"
    log(f"resolved_device={resolved_device}")
    if effective_passthrough_args:
        log(f"self_play_overrides={_format_command(effective_passthrough_args)}")
    else:
        log("No self-play overrides were provided. Using wrapper defaults.")

    self_play_command = None
    if args.skip_run:
        log("Skipping self-play execution and rebuilding manifest from existing episode files.")
    else:
        self_play_command = [
            sys.executable,
            "-u",
            str(SELF_PLAY_SCRIPT),
            "--model-path",
            str(source_model),
            "--output-dir",
            str(episodes_dir),
            *effective_passthrough_args,
        ]
        log(f"$ {_format_command(self_play_command)}")
        subprocess.run(self_play_command, check=True)
        log("Self-play subprocess completed.")

    episode_files = _discover_episode_files(episodes_dir)
    if not episode_files:
        raise FileNotFoundError(
            f"No episode files were found in {episodes_dir}. Nothing to export."
        )
    log(f"Discovered {len(episode_files)} episode file(s).")

    manifest = _build_manifest(
        args=args,
        batch_dir=batch_dir,
        episodes_dir=episodes_dir,
        source_model=source_model,
        source_model_sha256=source_model_sha256,
        self_play_command=self_play_command,
        passthrough_args=effective_passthrough_args,
        episode_files=episode_files,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log(f"manifest={manifest_path}")
    log(f"episode_count={manifest['summary']['episode_count']}")
    log(f"sample_count={manifest['summary']['sample_count']}")
    log(f"total_steps={manifest['summary']['total_steps']}")
    log(f"avg_steps_per_episode={manifest['summary']['avg_steps_per_episode']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

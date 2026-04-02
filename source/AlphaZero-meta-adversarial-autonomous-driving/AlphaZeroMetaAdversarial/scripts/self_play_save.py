from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

try:
    from AlphaZeroMetaAdversarial.core.settings import (
        ACTIVE_SCENARIO,
        CONFIG_PATH,
        SELF_PLAY_CONFIG,
    )
    from AlphaZeroMetaAdversarial.environment.config import build_env_spec
    from AlphaZeroMetaAdversarial.network.alphazero_network import AlphaZeroNetwork
    from AlphaZeroMetaAdversarial.training.trainer import AdversarialAlphaZeroTrainer
except ModuleNotFoundError:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))
    from AlphaZeroMetaAdversarial.core.settings import (
        ACTIVE_SCENARIO,
        CONFIG_PATH,
        SELF_PLAY_CONFIG,
    )
    from AlphaZeroMetaAdversarial.environment.config import build_env_spec
    from AlphaZeroMetaAdversarial.network.alphazero_network import AlphaZeroNetwork
    from AlphaZeroMetaAdversarial.training.trainer import AdversarialAlphaZeroTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "self_play"
DEFAULT_NETWORK_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run meta-adversarial AlphaZero self-play only and save the generated "
            "training shards plus manifest metadata."
        )
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episodes-per-shard", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=21)
    parser.add_argument("--env-seed", type=int, default=10)
    parser.add_argument("--episode-index-start", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--network-seed", type=int, default=DEFAULT_NETWORK_SEED)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--vehicles-count", type=int, default=None)
    parser.add_argument("--policy-frequency", type=int, default=None)
    parser.add_argument("--simulation-frequency", type=int, default=None)

    parser.add_argument("--n-simulations", type=int, default=None)
    parser.add_argument("--c-puct", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--temperature-drop-step", type=int, default=None)
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    parser.add_argument("--root-exploration-fraction", type=float, default=None)
    parser.add_argument("--max-expand-actions-per-agent", type=int, default=None)
    parser.add_argument("--no-reuse-tree-between-steps", action="store_true")
    parser.add_argument("--disable-root-dirichlet-noise", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    base_dir = DEFAULT_OUTPUT_ROOT / timestamp
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=False)
        return base_dir

    suffix = 1
    while True:
        candidate = DEFAULT_OUTPUT_ROOT / f"{timestamp}_{suffix:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def _build_config_from_args(args: argparse.Namespace):
    config = SELF_PLAY_CONFIG
    overrides = {}
    for field_name, value in (
        ("n_simulations", args.n_simulations),
        ("c_puct", args.c_puct),
        ("temperature", args.temperature),
        ("temperature_drop_step", args.temperature_drop_step),
        ("root_dirichlet_alpha", args.dirichlet_alpha),
        ("root_exploration_fraction", args.root_exploration_fraction),
        ("max_expand_actions_per_agent", args.max_expand_actions_per_agent),
    ):
        if value is not None:
            overrides[field_name] = value
    if overrides:
        config = replace(config, **overrides)
    return config


def _resolve_self_play_env_spec(args: argparse.Namespace):
    env_overrides = {}
    if args.duration is not None:
        env_overrides["duration"] = int(args.duration)
    if args.vehicles_count is not None:
        env_overrides["vehicles_count"] = int(args.vehicles_count)
    if args.policy_frequency is not None:
        env_overrides["policy_frequency"] = int(args.policy_frequency)
    if args.simulation_frequency is not None:
        env_overrides["simulation_frequency"] = int(args.simulation_frequency)

    return build_env_spec(
        stage="self_play",
        env_name=args.env_id,
        render_mode=None,
        env_config_overrides=env_overrides or None,
    )


def _build_network(config) -> AlphaZeroNetwork:
    return AlphaZeroNetwork(
        input_shape=config.input_shape,
        n_residual_layers=config.n_residual_layers,
        n_actions=config.n_actions,
        channels=config.network_channels,
        dropout_p=config.network_dropout_p,
        target_vector_dim=config.target_vector_dim,
        target_hidden_dim=config.target_hidden_dim,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_bootstrap_model(
    *,
    config,
    output_dir: Path,
    network_seed: int,
) -> tuple[Path, str]:
    model_path = output_dir / f"bootstrap_model_seed_{int(network_seed)}.pth"
    if not model_path.exists():
        torch.manual_seed(int(network_seed))
        network = _build_network(config)
        network.eval()
        torch.save(network.state_dict(), model_path)
    return model_path, _sha256_file(model_path)


def _serialize_examples(
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
):
    if not examples:
        return (
            torch.empty((0, 0, 0, 0), dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.float32),
            torch.empty((0, 1), dtype=torch.float32),
        )

    states, target_vectors, policies, values = zip(*examples)
    state_tensor = torch.from_numpy(np.stack(states, axis=0)).to(dtype=torch.float32)
    target_vector_tensor = torch.from_numpy(np.stack(target_vectors, axis=0)).to(
        dtype=torch.float32
    )
    policy_tensor = torch.from_numpy(np.stack(policies, axis=0)).to(dtype=torch.float32)
    value_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
    return (
        state_tensor,
        target_vector_tensor,
        policy_tensor,
        value_tensor,
    )


def _flush_shard(
    *,
    shard_examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    shard_episode_summaries: list[dict],
    shard_index: int,
    output_dir: Path,
    model_path: Path,
    model_sha256: str,
    network_seed: int,
) -> dict | None:
    if not shard_examples:
        return None

    states, target_vectors, policies, values = _serialize_examples(shard_examples)
    shard_path = output_dir / f"worker_00_shard_{shard_index:03d}.pt"
    torch.save(
        {
            "worker_id": 0,
            "shard_index": int(shard_index),
            "sample_count": int(states.shape[0]),
            "episode_count": len(shard_episode_summaries),
            "policy_format": "flat_meta",
            "network_seed": int(network_seed),
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "states": states,
            "target_vectors": target_vectors,
            "policies": policies,
            "values": values,
            "episodes": shard_episode_summaries,
        },
        shard_path,
    )
    print(
        f"[shard] shard={shard_index} episodes={len(shard_episode_summaries)} "
        f"samples={int(states.shape[0])} path={shard_path}",
        flush=True,
    )
    return {
        "worker_id": 0,
        "shard_index": int(shard_index),
        "sample_count": int(states.shape[0]),
        "episode_count": len(shard_episode_summaries),
        "path": str(shard_path),
    }


def _build_step_callback(progress_interval: int):
    if progress_interval <= 0:
        return None

    def _step_callback(info: dict) -> None:
        step = int(info["step"])
        done = bool(info["done"])
        if step % progress_interval != 0 and not done:
            return
        search_stats = info.get("search_stats") or {}
        print(
            f"[self-play] step={step} done={done} "
            f"rollouts={int(search_stats.get('rollouts', 0))} "
            f"rps={float(search_stats.get('rollouts_per_sec', 0.0)):.2f} "
            f"nn={float(search_stats.get('avg_inference_ms', 0.0)):.1f}ms",
            flush=True,
        )

    return _step_callback


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be a positive integer.")
    if args.episodes_per_shard <= 0:
        raise ValueError("--episodes-per-shard must be a positive integer.")

    output_dir = _resolve_output_dir(args)
    config = _build_config_from_args(args)
    env_spec = _resolve_self_play_env_spec(args)

    if args.model_path:
        model_path = Path(args.model_path).expanduser().resolve()
        model_source = "provided"
        model_sha256 = _sha256_file(model_path)
    else:
        model_path, model_sha256 = _ensure_bootstrap_model(
            config=config,
            output_dir=output_dir,
            network_seed=int(args.network_seed),
        )
        model_source = "bootstrap_seed"

    import gymnasium as gym
    import highway_env  # noqa: F401

    env = gym.make(
        env_spec.env_id,
        config=env_spec.config,
        render_mode=None,
    )
    torch.manual_seed(int(args.network_seed))
    network = _build_network(config)
    trainer = AdversarialAlphaZeroTrainer(
        network=network,
        config=config,
        env=env,
        device=args.device,
        verbose=not args.quiet,
        reuse_tree_between_steps=not bool(args.no_reuse_tree_between_steps),
        add_root_dirichlet_noise=not bool(args.disable_root_dirichlet_noise),
    )
    trainer.load_model(str(model_path))

    total_samples = 0
    shard_examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    shard_episode_summaries: list[dict] = []
    shard_manifests: list[dict] = []
    shard_index = 0

    print(
        "[meta-adversarial-self-play-save] "
        f"scenario={ACTIVE_SCENARIO} "
        f"config={CONFIG_PATH} "
        f"env_id={env_spec.env_id} "
        f"episodes={args.episodes} "
        f"device={args.device} "
        f"model={model_source}:{model_path} "
        f"network_seed={args.network_seed} "
        f"model_sha256={model_sha256} "
        f"output_dir={output_dir}",
        flush=True,
    )

    step_callback = _build_step_callback(int(args.progress_interval))

    try:
        for episode_offset in range(int(args.episodes)):
            episode_seed = int(args.seed_start) + episode_offset
            episode_index = int(args.episode_index_start) + episode_offset
            started_at = time.perf_counter()
            summary = trainer.run_episode(
                seed=episode_seed,
                episode_index=episode_index,
                max_steps=args.max_steps_per_episode,
                store_in_replay=False,
                return_examples=True,
                add_root_dirichlet_noise=not bool(args.disable_root_dirichlet_noise),
                sample_actions=True,
                step_callback=step_callback,
            )
            elapsed = time.perf_counter() - started_at
            episode_examples = summary.pop("episode_examples")
            sample_count = len(episode_examples)
            total_samples += sample_count
            shard_examples.extend(episode_examples)
            summary["sample_count"] = int(sample_count)
            summary["elapsed_s"] = float(elapsed)
            summary["device"] = str(trainer.device)
            summary["network_seed"] = int(args.network_seed)
            shard_episode_summaries.append(summary)

            print(
                f"[episode] index={episode_index} seed={episode_seed} "
                f"steps={summary['steps']} samples={sample_count} "
                f"outcome={summary['outcome_reason']} time={elapsed:.2f}s",
                flush=True,
            )

            if len(shard_episode_summaries) >= int(args.episodes_per_shard):
                shard_manifest = _flush_shard(
                    shard_examples=shard_examples,
                    shard_episode_summaries=shard_episode_summaries,
                    shard_index=shard_index,
                    output_dir=output_dir,
                    model_path=model_path,
                    model_sha256=model_sha256,
                    network_seed=int(args.network_seed),
                )
                if shard_manifest is not None:
                    shard_manifests.append(shard_manifest)
                shard_examples = []
                shard_episode_summaries = []
                shard_index += 1

        shard_manifest = _flush_shard(
            shard_examples=shard_examples,
            shard_episode_summaries=shard_episode_summaries,
            shard_index=shard_index,
            output_dir=output_dir,
            model_path=model_path,
            model_sha256=model_sha256,
            network_seed=int(args.network_seed),
        )
        if shard_manifest is not None:
            shard_manifests.append(shard_manifest)
    finally:
        env.close()

    manifest = {
        "created_at_epoch_s": time.time(),
        "active_scenario": ACTIVE_SCENARIO,
        "config_path": str(CONFIG_PATH),
        "env_id": env_spec.env_id,
        "scenario_render_mode": env_spec.render_mode,
        "worker_render_mode": None,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "config": asdict(config),
        "model_path": str(model_path),
        "model_source": model_source,
        "model_sha256": model_sha256,
        "network_seed": int(args.network_seed),
        "device": str(trainer.device),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "total_episodes": int(args.episodes),
        "total_samples": int(total_samples),
        "total_shards": int(len(shard_manifests)),
        "shards": shard_manifests,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "model_path": str(model_path),
                "model_sha256": model_sha256,
                "total_samples": manifest["total_samples"],
                "total_shards": manifest["total_shards"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

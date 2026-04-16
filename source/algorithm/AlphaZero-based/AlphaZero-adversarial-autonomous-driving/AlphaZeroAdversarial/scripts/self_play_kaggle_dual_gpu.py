from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from queue import Empty

import numpy as np
import torch

try:
    from AlphaZeroAdversarial.core.settings import (
        ACTIVE_SCENARIO,
        CONFIG_PATH,
        SELF_PLAY_CONFIG,
    )
    from AlphaZeroAdversarial.environment.config import build_env_spec
    from AlphaZeroAdversarial.network.alphazero_network import AlphaZeroNetwork
    from AlphaZeroAdversarial.training.trainer import AdversarialAlphaZeroTrainer
except ModuleNotFoundError:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))
    from AlphaZeroAdversarial.core.settings import (
        ACTIVE_SCENARIO,
        CONFIG_PATH,
        SELF_PLAY_CONFIG,
    )
    from AlphaZeroAdversarial.environment.config import build_env_spec
    from AlphaZeroAdversarial.network.alphazero_network import AlphaZeroNetwork
    from AlphaZeroAdversarial.training.trainer import AdversarialAlphaZeroTrainer


DEFAULT_OUTPUT_DIR = Path("/kaggle/working/alphazero_adversarial_self_play")


def _parse_gpu_indices(raw_value: str | None) -> list[int] | None:
    if raw_value is None:
        return None
    values: list[int] = []
    for chunk in raw_value.split(","):
        item = chunk.strip()
        if not item:
            continue
        index = int(item)
        if index < 0:
            raise ValueError("GPU indices must be non-negative integers.")
        if index not in values:
            values.append(index)
    return values or None


def _get_worker_devices(args: argparse.Namespace) -> list[str]:
    requested_device = str(args.device).lower()
    if requested_device == "cpu":
        return ["cpu"] * int(args.workers)

    if requested_device.startswith("cuda:"):
        return [requested_device] * int(args.workers)

    if requested_device not in {"auto", "cuda"}:
        return [str(args.device)] * int(args.workers)

    selected_indices = _parse_gpu_indices(args.gpu_indices)
    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if visible_gpu_count == 0:
        if requested_device == "cuda":
            raise RuntimeError("No CUDA devices are available.")
        return ["cpu"] * int(args.workers)

    candidate_indices = (
        list(range(visible_gpu_count))
        if selected_indices is None
        else list(selected_indices)
    )
    for index in candidate_indices:
        if index >= visible_gpu_count:
            raise ValueError(
                f"Requested GPU index {index}, but only {visible_gpu_count} visible CUDA device(s) exist."
            )

    if args.num_gpus is not None:
        requested_gpu_count = int(args.num_gpus)
        if requested_gpu_count <= 0:
            raise ValueError("--num-gpus must be a positive integer.")
        candidate_indices = candidate_indices[:requested_gpu_count]

    if not candidate_indices:
        if requested_device == "cuda":
            raise RuntimeError("No CUDA devices are selected.")
        return ["cpu"] * int(args.workers)

    worker_count = int(args.workers)
    return [
        f"cuda:{candidate_indices[worker_index % len(candidate_indices)]}"
        for worker_index in range(worker_count)
    ]


def _get_worker_episode_counts(args: argparse.Namespace) -> list[int]:
    worker_count = int(args.workers)
    if worker_count <= 0:
        raise ValueError("--workers must be a positive integer.")

    if args.total_episodes is not None:
        total_episodes = int(args.total_episodes)
        if total_episodes <= 0:
            raise ValueError("--total-episodes must be a positive integer.")
        base_count, remainder = divmod(total_episodes, worker_count)
        return [
            base_count + (1 if worker_id < remainder else 0)
            for worker_id in range(worker_count)
        ]

    episodes_per_worker = int(args.episodes_per_worker)
    if episodes_per_worker <= 0:
        raise ValueError("--episodes-per-worker must be a positive integer.")
    return [episodes_per_worker] * worker_count


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


def _get_episode_mode(args: argparse.Namespace, worker_episode_counts: list[int]) -> str:
    if args.total_episodes is not None:
        return f"total_episodes={int(sum(worker_episode_counts))}"
    return f"episodes_per_worker={int(args.episodes_per_worker)}"


def _build_self_play_env_spec(args: argparse.Namespace):
    env_overrides = {}
    if args.duration is not None:
        env_overrides["duration"] = int(args.duration)
    if args.other_vehicles is not None:
        env_overrides["other_vehicles"] = int(args.other_vehicles)
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


def _serialize_examples(
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]
):
    if not examples:
        return (
            torch.empty((0, 0, 0, 0), dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.float32),
            torch.empty((0, 1), dtype=torch.float32),
        )

    states, target_vectors, accelerate_policies, steering_policies, values = zip(*examples)
    state_tensor = torch.from_numpy(np.stack(states, axis=0)).to(dtype=torch.float32)
    target_vector_tensor = torch.from_numpy(np.stack(target_vectors, axis=0)).to(
        dtype=torch.float32
    )
    accelerate_policy_tensor = torch.from_numpy(
        np.stack(accelerate_policies, axis=0)
    ).to(dtype=torch.float32)
    steering_policy_tensor = torch.from_numpy(
        np.stack(steering_policies, axis=0)
    ).to(dtype=torch.float32)
    value_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
    return (
        state_tensor,
        target_vector_tensor,
        accelerate_policy_tensor,
        steering_policy_tensor,
        value_tensor,
    )


def _flush_shard(
    *,
    shard_examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
    shard_episode_summaries: list[dict],
    worker_id: int,
    shard_index: int,
    output_dir: Path,
) -> dict | None:
    if not shard_examples:
        return None

    states, target_vectors, accelerate_policies, steering_policies, values = _serialize_examples(
        shard_examples
    )
    shard_path = output_dir / f"worker_{worker_id:02d}_shard_{shard_index:03d}.pt"
    torch.save(
        {
            "worker_id": int(worker_id),
            "shard_index": int(shard_index),
            "sample_count": int(states.shape[0]),
            "episode_count": len(shard_episode_summaries),
            "policy_format": "factorized_axes",
            "states": states,
            "target_vectors": target_vectors,
            "accelerate_policies": accelerate_policies,
            "steering_policies": steering_policies,
            "values": values,
            "episodes": shard_episode_summaries,
        },
        shard_path,
    )
    print(
        f"[shard] worker={worker_id} shard={shard_index} "
        f"episodes={len(shard_episode_summaries)} samples={int(states.shape[0])} "
        f"path={shard_path}",
        flush=True,
    )
    return {
        "worker_id": int(worker_id),
        "shard_index": int(shard_index),
        "sample_count": int(states.shape[0]),
        "episode_count": len(shard_episode_summaries),
        "path": str(shard_path),
    }


def _run_worker(task: dict) -> dict:
    worker_id = int(task["worker_id"])
    device = str(task["device"])
    if device.startswith("cuda:"):
        torch.cuda.set_device(device)

    torch.set_num_threads(int(task["torch_threads_per_worker"]))
    torch.manual_seed(int(task["network_seed"]))
    np.random.seed(int(task["self_play_seed"]))

    import gymnasium as gym
    import highway_env

    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        task["env_id"],
        config=task["env_config"],
        render_mode=task["render_mode"],
    )
    network = AlphaZeroNetwork(
        input_shape=tuple(task["input_shape"]),
        n_residual_layers=int(task["n_residual_layers"]),
        n_actions=int(task["n_actions"]),
        n_action_axis_0=int(task["n_action_axis_0"]),
        n_action_axis_1=int(task["n_action_axis_1"]),
        channels=int(task["network_channels"]),
        dropout_p=float(task["network_dropout_p"]),
        target_vector_dim=int(task["target_vector_dim"]),
        target_hidden_dim=int(task["target_hidden_dim"]),
    )
    network.load_state_dict(
        torch.load(task["model_path"], map_location=torch.device("cpu"))
    )
    trainer = AdversarialAlphaZeroTrainer(
        network=network,
        config=task["config"],
        env=env,
        device=device,
        verbose=False,
        reuse_tree_between_steps=bool(task["reuse_tree_between_steps"]),
        add_root_dirichlet_noise=bool(task["add_root_dirichlet_noise"]),
    )

    progress_interval = int(task["progress_interval"])
    episodes_per_worker = int(task["episodes_per_worker"])
    max_steps_per_episode = task["max_steps_per_episode"]
    if max_steps_per_episode is not None:
        max_steps_per_episode = int(max_steps_per_episode)
    episodes_per_shard = int(task["episodes_per_shard"])

    shard_examples: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
    ] = []
    shard_episode_summaries: list[dict] = []
    shard_manifests: list[dict] = []
    shard_index = 0
    total_samples = 0

    print(
        f"[worker {worker_id}] pid={os.getpid()} device={device} "
        f"episodes={episodes_per_worker} n_simulations={task['config'].n_simulations} "
        f"reuse_tree={int(task['reuse_tree_between_steps'])} "
        f"output={output_dir}",
        flush=True,
    )
    print(
        f"[worker {worker_id}] highway_env={Path(highway_env.__file__).resolve()}",
        flush=True,
    )

    try:
        for episode_offset in range(episodes_per_worker):
            episode_seed = int(task["self_play_seed"]) + episode_offset
            episode_index = int(task["episode_index_offset"]) + episode_offset
            started_at = time.perf_counter()

            def _step_callback(info: dict) -> None:
                if progress_interval <= 0:
                    return
                step = int(info["step"])
                done = bool(info["done"])
                if step % progress_interval != 0 and not done:
                    return
                search_stats = info.get("search_stats") or {}
                print(
                    f"[worker {worker_id}] episode={episode_index} step={step} done={done} "
                    f"rollouts={int(search_stats.get('rollouts', 0))} "
                    f"rps={float(search_stats.get('rollouts_per_sec', 0.0)):.2f} "
                    f"nn={float(search_stats.get('avg_inference_ms', 0.0)):.1f}ms",
                    flush=True,
                )

            summary = trainer.run_episode(
                seed=episode_seed,
                episode_index=episode_index,
                max_steps=max_steps_per_episode,
                store_in_replay=False,
                return_examples=True,
                add_root_dirichlet_noise=bool(task["add_root_dirichlet_noise"]),
                sample_actions=True,
                step_callback=_step_callback,
            )
            elapsed = time.perf_counter() - started_at
            episode_examples = summary.pop("episode_examples")
            sample_count = len(episode_examples)
            total_samples += sample_count
            shard_examples.extend(episode_examples)
            summary["sample_count"] = int(sample_count)
            summary["device"] = device
            summary["elapsed_s"] = float(elapsed)
            shard_episode_summaries.append(summary)

            print(
                f"[worker {worker_id}] episode={episode_index} "
                f"steps={summary['steps']} samples={sample_count} "
                f"outcome={summary['outcome_reason']} time={elapsed:.2f}s",
                flush=True,
            )

            if len(shard_episode_summaries) >= episodes_per_shard:
                shard_manifest = _flush_shard(
                    shard_examples=shard_examples,
                    shard_episode_summaries=shard_episode_summaries,
                    worker_id=worker_id,
                    shard_index=shard_index,
                    output_dir=output_dir,
                )
                if shard_manifest is not None:
                    shard_manifests.append(shard_manifest)
                shard_examples = []
                shard_episode_summaries = []
                shard_index += 1

        shard_manifest = _flush_shard(
            shard_examples=shard_examples,
            shard_episode_summaries=shard_episode_summaries,
            worker_id=worker_id,
            shard_index=shard_index,
            output_dir=output_dir,
        )
        if shard_manifest is not None:
            shard_manifests.append(shard_manifest)
    finally:
        env.close()

    return {
        "worker_id": worker_id,
        "device": device,
        "episodes_per_worker": episodes_per_worker,
        "total_samples": total_samples,
        "shards": shard_manifests,
    }


def _worker_entry(task: dict, result_queue) -> None:
    worker_id = int(task["worker_id"])
    try:
        result_queue.put({"ok": True, "result": _run_worker(task)})
    except Exception:
        result_queue.put(
            {
                "ok": False,
                "worker_id": worker_id,
                "error": traceback.format_exc(),
            }
        )


def _terminate_live_processes(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            print(
                f"[manager] terminating worker pid={process.pid}",
                flush=True,
            )
            process.terminate()

    for process in processes:
        process.join(timeout=5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adversarial AlphaZero self-play on Kaggle with one worker per GPU."
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--episodes-per-worker", type=int, default=2)
    parser.add_argument("--total-episodes", type=int, default=None)
    parser.add_argument("--episodes-per-shard", type=int, default=2)
    parser.add_argument("--self-play-seed", type=int, default=1000)
    parser.add_argument("--network-seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--gpu-indices", type=str, default="0,1")
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--result-timeout", type=float, default=0.0)

    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--other-vehicles", type=int, default=None)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config_from_args(args)
    env_spec = _build_self_play_env_spec(args)
    worker_devices = _get_worker_devices(args)
    worker_episode_counts = _get_worker_episode_counts(args)
    episode_mode = _get_episode_mode(args, worker_episode_counts)

    if args.episodes_per_shard <= 0:
        raise ValueError("--episodes-per-shard must be a positive integer.")

    model_path = None
    created_temp_model = False
    if args.model_path:
        model_path = Path(args.model_path).expanduser().resolve()
    else:
        torch.manual_seed(int(args.network_seed))
        network = AlphaZeroNetwork(
            input_shape=config.input_shape,
            n_residual_layers=config.n_residual_layers,
            n_actions=config.n_actions,
            n_action_axis_0=config.n_action_axis_0,
            n_action_axis_1=config.n_action_axis_1,
            channels=config.network_channels,
            dropout_p=config.network_dropout_p,
            target_vector_dim=config.target_vector_dim,
            target_hidden_dim=config.target_hidden_dim,
        )
        model_path = output_dir / "_initial_model_for_kaggle_parallel_self_play.pth"
        torch.save(network.state_dict(), model_path)
        created_temp_model = True

    model_display = str(model_path) if args.model_path else f"bootstrap:{model_path}"
    print(
        "[adversarial-self-play-kaggle-dual-gpu] "
        f"scenario={ACTIVE_SCENARIO} "
        f"config={CONFIG_PATH} "
        f"env_id={env_spec.env_id} "
        f"model={model_display} "
        f"workers={args.workers} "
        f"{episode_mode} "
        f"device={args.device} "
        f"n_simulations={config.n_simulations} "
        f"reuse_tree={int(not args.no_reuse_tree_between_steps)} "
        f"num_gpus={args.num_gpus if args.num_gpus is not None else 'auto'} "
        f"gpu_indices={args.gpu_indices if args.gpu_indices is not None else 'auto'} "
        f"result_timeout={args.result_timeout if args.result_timeout > 0 else 'disabled'}",
        flush=True,
    )
    print(f"config_path={CONFIG_PATH}", flush=True)
    print(f"active_scenario={ACTIVE_SCENARIO}", flush=True)
    print(f"env_id={env_spec.env_id}", flush=True)
    print(f"workers={args.workers} worker_devices={worker_devices}", flush=True)
    print(f"worker_episode_counts={worker_episode_counts}", flush=True)
    print(f"output_dir={output_dir}", flush=True)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes: list[mp.Process] = []
    next_episode_offset = 0

    for worker_id, (device, episode_count) in enumerate(
        zip(worker_devices, worker_episode_counts, strict=True)
    ):
        if int(episode_count) <= 0:
            continue
        task = {
            "worker_id": worker_id,
            "device": device,
            "episodes_per_worker": int(episode_count),
            "episode_index_offset": int(next_episode_offset),
            "episodes_per_shard": int(args.episodes_per_shard),
            "self_play_seed": int(args.self_play_seed) + worker_id * 10000,
            "network_seed": int(args.network_seed),
            "torch_threads_per_worker": int(args.torch_threads_per_worker),
            "max_steps_per_episode": args.max_steps_per_episode,
            "progress_interval": int(args.progress_interval),
            "output_dir": str(output_dir),
            "model_path": str(model_path),
            "render_mode": None,
            "env_id": env_spec.env_id,
            "env_config": env_spec.config,
            "input_shape": tuple(config.input_shape),
            "n_actions": int(config.n_actions),
            "n_action_axis_0": int(config.n_action_axis_0),
            "n_action_axis_1": int(config.n_action_axis_1),
            "n_residual_layers": int(config.n_residual_layers),
            "network_channels": int(config.network_channels),
            "network_dropout_p": float(config.network_dropout_p),
            "target_vector_dim": int(config.target_vector_dim),
            "target_hidden_dim": int(config.target_hidden_dim),
            "config": config,
            "reuse_tree_between_steps": not bool(args.no_reuse_tree_between_steps),
            "add_root_dirichlet_noise": True,
        }
        process = ctx.Process(
            target=_worker_entry,
            args=(task, result_queue),
            name=f"adversarial-self-play-worker-{worker_id}",
        )
        process.start()
        processes.append(process)
        print(
            f"[manager] launched worker={worker_id} pid={process.pid} "
            f"device={device} episodes={episode_count}",
            flush=True,
        )
        next_episode_offset += int(episode_count)

    pending_worker_ids = set(range(len(processes)))
    worker_results = []

    try:
        while pending_worker_ids:
            timeout = None if args.result_timeout <= 0 else max(1.0, float(args.result_timeout))
            try:
                result = result_queue.get(timeout=timeout)
            except Empty:
                pending_alive = [
                    worker_id
                    for worker_id in sorted(pending_worker_ids)
                    if processes[worker_id].is_alive()
                ]
                if pending_alive:
                    raise TimeoutError(
                        "Timed out waiting for worker results. "
                        f"Still running: {pending_alive}"
                    )
                raise RuntimeError("Timed out and found no live workers.")

            if not result["ok"]:
                raise RuntimeError(
                    f"Worker {result['worker_id']} failed:\n{result['error']}"
                )

            worker_result = result["result"]
            pending_worker_ids.discard(int(worker_result["worker_id"]))
            worker_results.append(worker_result)
            print(
                f"[manager] worker={worker_result['worker_id']} device={worker_result['device']} "
                f"samples={worker_result['total_samples']} shards={len(worker_result['shards'])}",
                flush=True,
            )

        for process in processes:
            process.join(timeout=5.0)
            if process.exitcode not in (0, None):
                raise RuntimeError(
                    f"Worker process {process.pid} exited with code {process.exitcode}."
                )
    except Exception:
        _terminate_live_processes(processes)
        raise

    manifest = {
        "created_at_epoch_s": time.time(),
        "config_path": str(CONFIG_PATH),
        "active_scenario": ACTIVE_SCENARIO,
        "env_id": env_spec.env_id,
        "scenario_render_mode": env_spec.render_mode,
        "worker_render_mode": None,
        "worker_devices": worker_devices,
        "worker_episode_counts": worker_episode_counts,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "config": asdict(config),
        "model_path": str(model_path),
        "total_samples": int(sum(item["total_samples"] for item in worker_results)),
        "total_shards": int(sum(len(item["shards"]) for item in worker_results)),
        "workers": sorted(worker_results, key=lambda item: int(item["worker_id"])),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "total_samples": manifest["total_samples"],
                "total_shards": manifest["total_shards"],
                "workers": len(worker_results),
            },
            indent=2,
        ),
        flush=True,
    )

    if created_temp_model:
        try:
            model_path.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

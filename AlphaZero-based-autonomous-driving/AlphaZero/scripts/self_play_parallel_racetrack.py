import argparse
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Empty

import gymnasium as gym
import numpy as np
import torch

import highway_env  # noqa: F401

try:
    from core.settings import SELF_PLAY_CONFIG
    from network.alphazero_network import AlphaZeroNetwork
    from training.trainer import AlphaZeroTrainer
except ModuleNotFoundError as exc:
    if exc.name not in {"core", "network", "training", "AlphaZero"}:
        raise
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from core.settings import SELF_PLAY_CONFIG
    from network.alphazero_network import AlphaZeroNetwork
    from training.trainer import AlphaZeroTrainer


def build_racetrack_env_config(
    duration: int,
    other_vehicles: int,
    finish_laps: int,
    terminate_on_finish: bool,
) -> dict:
    return {
        "observation": {
            "type": "DetailedOccupancyGrid",
            "features": ["presence", "on_lane", "on_road"],
            "grid_size": [[-50, 50], [-12, 12]],
            "grid_step": [1.0, 1.0],
            "absolute": False,
            "align_to_vehicle_axes": True,
            "include_ego_vehicle": True,
            "on_road_mode": "area",
            "on_road_soft_mode": True,
            "presence_subsamples": 5,
            "on_road_subsamples": 3,
        },
        "action": {
            "type": "DiscreteAction",
            "longitudinal": True,
            "lateral": True,
            "actions_per_axis": 5,
            "acceleration_range": [-5.0, 5.0],
            "steering_range": [-np.pi / 6, np.pi / 6],
        },
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "duration": duration,
        "collision_reward": -1,
        "lane_centering_cost": 4,
        "lane_centering_reward": 1,
        # RacetrackEnv computes np.linalg.norm(action). With discrete ids this term
        # should be neutralized to avoid reward distortion from action index values.
        "action_reward": 0.0,
        "controlled_vehicles": 1,
        "other_vehicles": other_vehicles,
        "terminate_off_road": True,
        "terminate_on_finish": terminate_on_finish,
        "finish_laps": finish_laps,
        "finish_line_segment": ("a", "b"),
        "finish_line_longitudinal": 0.0,
    }


def _serialize_training_data(training_data, n_actions: int):
    if not training_data:
        return (
            torch.empty((0, 0, 0, 0), dtype=torch.float32),
            torch.empty((0, n_actions), dtype=torch.float32),
            torch.empty((0, 1), dtype=torch.float32),
        )

    states, policies, values = zip(*training_data)
    state_tensor = torch.cat(states)
    policy_tensor = torch.tensor(
        [[policy.get(action, 0.0) for action in range(n_actions)] for policy in policies],
        dtype=torch.float32,
    )
    value_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
    return state_tensor, policy_tensor, value_tensor


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m{int(remaining_seconds):02d}s"
    hours, remaining_minutes = divmod(int(minutes), 60)
    return f"{hours}h{remaining_minutes:02d}m"


def _init_search_stats_accumulator() -> dict:
    return {
        "steps": 0,
        "search_time_s": 0.0,
        "rollouts": 0,
        "rollout_time_s": 0.0,
        "inference_calls": 0,
        "inference_time_s": 0.0,
    }


def _accumulate_search_stats(accumulator: dict, search_stats: dict | None) -> None:
    if not search_stats:
        return
    accumulator["steps"] += 1
    accumulator["search_time_s"] += float(search_stats.get("search_time_s", 0.0))
    accumulator["rollouts"] += int(search_stats.get("rollouts", 0))
    accumulator["rollout_time_s"] += float(search_stats.get("rollout_time_s", 0.0))
    accumulator["inference_calls"] += int(search_stats.get("inference_calls", 0))
    accumulator["inference_time_s"] += float(search_stats.get("inference_time_s", 0.0))


def _format_step_search_stats(search_stats: dict | None) -> str:
    if not search_stats:
        return "mcts=unavailable"
    return (
        f"mcts_rollouts={int(search_stats.get('rollouts', 0))} "
        f"search={float(search_stats.get('search_time_s', 0.0)):.2f}s "
        f"mcts_rps={float(search_stats.get('effective_rollouts_per_sec', 0.0)):.2f} "
        f"nn_calls={int(search_stats.get('inference_calls', 0))} "
        f"nn_avg={float(search_stats.get('avg_inference_ms', 0.0)):.1f}ms/state"
    )


def _format_episode_search_summary(accumulator: dict) -> str:
    decisions = int(accumulator["steps"])
    rollouts = int(accumulator["rollouts"])
    inference_calls = int(accumulator["inference_calls"])
    search_time_s = float(accumulator["search_time_s"])
    rollout_time_s = float(accumulator["rollout_time_s"])
    inference_time_s = float(accumulator["inference_time_s"])
    avg_search_s = 0.0 if decisions == 0 else search_time_s / decisions
    avg_rollouts = 0.0 if decisions == 0 else rollouts / decisions
    avg_inference_ms = 0.0 if inference_calls == 0 else 1000.0 * inference_time_s / inference_calls
    avg_rollout_ms = 0.0 if rollouts == 0 else 1000.0 * rollout_time_s / rollouts
    effective_rollouts_per_sec = 0.0 if search_time_s <= 0.0 else rollouts / search_time_s
    return (
        f"decisions={decisions} avg_search={avg_search_s:.2f}s "
        f"avg_rollouts={avg_rollouts:.1f} mcts_rps={effective_rollouts_per_sec:.2f} "
        f"nn_avg={avg_inference_ms:.1f}ms/state avg_rollout={avg_rollout_ms:.1f}ms"
    )


def _run_worker(task: dict) -> dict:
    torch.set_num_threads(int(task["torch_threads_per_worker"]))
    torch.manual_seed(int(task["network_seed"]))
    np.random.seed(int(task["self_play_seed"]))

    env = gym.make(task["env_id"], config=task["env_config"], render_mode="rgb_array")
    env.reset(seed=int(task["env_seed"]))

    network = AlphaZeroNetwork(
        input_shape=tuple(task["input_shape"]),
        n_residual_layers=int(task["n_residual_layers"]),
        n_actions=int(task["n_actions"]),
    )
    network.load_state_dict(torch.load(task["model_path"], map_location=torch.device("cpu")))

    trainer = AlphaZeroTrainer(
        network=network,
        env=env,
        c_puct=float(task["c_puct"]),
        n_simulations=int(task["n_simulations"]),
        learning_rate=float(task["learning_rate"]),
        weight_decay=float(task["weight_decay"]),
        stack_config=task["stack_config"],
        n_actions=int(task["n_actions"]),
        verbose=False,
        device=task["device"],
        max_root_visits=task["mcts_max_root_visits"],
        temperature=float(task["temperature"]),
        temperature_drop_step=task["temperature_drop_step"],
        add_root_dirichlet_noise=bool(task["add_root_dirichlet_noise"]),
        root_dirichlet_alpha=float(task["root_dirichlet_alpha"]),
        root_exploration_fraction=float(task["root_exploration_fraction"]),
    )

    worker_id = int(task["worker_id"])
    episodes_per_worker = int(task["episodes_per_worker"])
    n_actions = int(task["n_actions"])
    print_actions = bool(task["print_actions"])
    progress_interval = int(task["progress_interval"])
    worker_heartbeat_interval = float(task["worker_heartbeat_interval"])
    max_steps_per_episode = task["max_steps_per_episode"]
    max_steps_per_episode = (
        None if max_steps_per_episode is None else int(max_steps_per_episode)
    )
    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_state = {
        "episode_index": 0,
        "step": 0,
        "episode_seed": None,
        "episode_started_at": time.perf_counter(),
        "status": "booting",
    }
    worker_state_lock = threading.Lock()
    heartbeat_stop = threading.Event()

    def _worker_heartbeat_loop() -> None:
        while not heartbeat_stop.wait(worker_heartbeat_interval):
            with worker_state_lock:
                episode_index = int(worker_state["episode_index"])
                step = int(worker_state["step"])
                episode_seed = worker_state["episode_seed"]
                episode_started_at = float(worker_state["episode_started_at"])
                status = str(worker_state["status"])
            elapsed = time.perf_counter() - episode_started_at
            print(
                f"[worker {worker_id}] heartbeat "
                f"episode={episode_index + 1}/{episodes_per_worker} "
                f"step={step} elapsed={_format_duration(elapsed)} "
                f"seed={episode_seed} status={status} device={trainer.device}",
                flush=True,
            )

    heartbeat_thread = None
    if worker_heartbeat_interval > 0:
        heartbeat_thread = threading.Thread(
            target=_worker_heartbeat_loop,
            name=f"worker-{worker_id}-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()

    print(
        f"[worker {worker_id}] pid={os.getpid()} "
        f"episodes={episodes_per_worker} n_simulations={task['n_simulations']} "
        f"device={trainer.device}",
        flush=True,
    )

    episode_summaries = []
    total_samples = 0

    try:
        for episode_idx in range(episodes_per_worker):
            trainer.training_data.clear()
            trainer.action_list.clear()
            trainer.verbose = print_actions
            episode_search_stats = _init_search_stats_accumulator()

            episode_seed = int(task["self_play_seed"]) + episode_idx
            with worker_state_lock:
                worker_state["episode_index"] = episode_idx
                worker_state["step"] = 0
                worker_state["episode_seed"] = episode_seed
                worker_state["episode_started_at"] = time.perf_counter()
                worker_state["status"] = "starting"
            print(
                f"[worker {worker_id}] episode {episode_idx + 1}/{episodes_per_worker} "
                f"start seed={episode_seed}",
                flush=True,
            )
            t0 = time.perf_counter()

            def _step_callback(info: dict):
                step = int(info["step"])
                done = bool(info["done"])
                search_stats = info.get("search_stats")
                with worker_state_lock:
                    worker_state["step"] = step
                    worker_state["status"] = "done" if done else "running"
                _accumulate_search_stats(episode_search_stats, search_stats)
                if progress_interval > 0 and (step % progress_interval == 0 or done):
                    print(
                        f"[worker {worker_id}] episode {episode_idx + 1}/{episodes_per_worker} "
                        f"step={step} action={info['action']} done={done} "
                        f"{_format_step_search_stats(search_stats)}",
                        flush=True,
                    )

            trainer.self_play(
                seed=episode_seed,
                max_steps=max_steps_per_episode,
                step_callback=_step_callback,
            )
            elapsed = time.perf_counter() - t0

            states, policies, values = _serialize_training_data(trainer.training_data, n_actions=n_actions)
            sample_count = int(states.shape[0])
            total_samples += sample_count

            episode_path = output_dir / f"worker_{worker_id:02d}_episode_{episode_idx:03d}.pt"
            torch.save(
                {
                    "worker_id": worker_id,
                    "episode_index": episode_idx,
                    "self_play_seed": episode_seed,
                    "states": states,
                    "policies": policies,
                    "values": values,
                    "actions": list(trainer.action_list),
                },
                episode_path,
            )
            episode_summaries.append(
                {
                    "episode_index": episode_idx,
                    "sample_count": sample_count,
                    "path": str(episode_path),
                }
            )
            with worker_state_lock:
                worker_state["status"] = "saved"
            print(
                f"[worker {worker_id}] episode {episode_idx + 1}/{episodes_per_worker} "
                f"done samples={sample_count} steps={len(trainer.action_list)} "
                f"time={elapsed:.2f}s saved={episode_path.name} "
                f"{_format_episode_search_summary(episode_search_stats)}",
                flush=True,
            )
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)
        env.close()

    return {
        "worker_id": worker_id,
        "total_samples": total_samples,
        "episode_summaries": episode_summaries,
    }


def _worker_entry(task: dict, result_queue):
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


def _parse_gpu_indices(raw_value: str | None) -> list[int] | None:
    if raw_value is None:
        return None

    values = []
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


def _capability_tag(major: int, minor: int) -> str:
    return f"sm_{major}{minor}"


def _discover_supported_cuda_devices(
    selected_indices: list[int] | None,
) -> tuple[list[int], list[dict], int]:
    if not torch.cuda.is_available():
        return [], [], 0

    total_devices = torch.cuda.device_count()
    candidate_indices = (
        list(range(total_devices))
        if selected_indices is None
        else list(selected_indices)
    )
    arch_list = {str(item) for item in torch.cuda.get_arch_list()}
    supported_indices = []
    unsupported_devices = []

    for index in candidate_indices:
        if index >= total_devices:
            raise ValueError(
                f"Requested CUDA device index {index}, but only {total_devices} visible device(s) exist."
            )
        properties = torch.cuda.get_device_properties(index)
        capability = _capability_tag(properties.major, properties.minor)
        if arch_list and capability not in arch_list:
            unsupported_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": capability,
                }
            )
            continue
        supported_indices.append(index)

    return supported_indices, unsupported_devices, total_devices


def _resolve_worker_devices(args: argparse.Namespace) -> list[str]:
    requested_device = str(args.device).lower()
    gpu_indices = _parse_gpu_indices(args.gpu_indices)
    num_gpus = None if args.num_gpus is None else int(args.num_gpus)
    if num_gpus is not None and num_gpus <= 0:
        raise ValueError("--num-gpus must be a positive integer.")

    if requested_device == "cpu":
        if gpu_indices is not None or num_gpus is not None:
            print(
                "Ignoring --gpu-indices/--num-gpus because --device=cpu was requested.",
                flush=True,
            )
        return ["cpu"] * int(args.workers)

    if requested_device.startswith("cuda:"):
        if gpu_indices is not None or num_gpus is not None:
            raise ValueError(
                "--gpu-indices and --num-gpus cannot be combined with an explicit device like cuda:0."
            )
        return [requested_device] * int(args.workers)

    if requested_device not in {"auto", "cuda"}:
        if gpu_indices is not None or num_gpus is not None:
            raise ValueError(
                "--gpu-indices and --num-gpus are supported only with --device=auto or --device=cuda."
            )
        return [args.device] * int(args.workers)

    supported_indices, unsupported_devices, total_devices = _discover_supported_cuda_devices(
        gpu_indices
    )
    if num_gpus is not None:
        supported_indices = supported_indices[:num_gpus]

    if unsupported_devices:
        skipped = ", ".join(
            f"cuda:{item['index']} ({item['name']} {item['capability']})"
            for item in unsupported_devices
        )
        print(f"Skipping incompatible CUDA devices: {skipped}", flush=True)

    if not supported_indices:
        if requested_device == "auto":
            if total_devices > 0:
                print(
                    "No compatible CUDA devices available for this PyTorch build. Falling back to CPU.",
                    flush=True,
                )
            return ["cpu"] * int(args.workers)
        raise RuntimeError("No compatible CUDA devices available for --device=cuda.")

    worker_count = int(args.workers)
    return [f"cuda:{supported_indices[index % len(supported_indices)]}" for index in range(worker_count)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel self-play for racetrack-v0.")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel worker processes.")
    parser.add_argument(
        "--episodes-per-worker",
        type=int,
        default=2,
        help="Episodes generated by each worker.",
    )
    parser.add_argument("--env-id", default="racetrack-v0")
    parser.add_argument("--duration", type=int, default=80)
    parser.add_argument("--other-vehicles", type=int, default=1)
    parser.add_argument("--finish-laps", type=int, default=1)
    parser.add_argument(
        "--terminate-on-finish",
        dest="terminate_on_finish",
        action="store_true",
        default=True,
        help="End the episode as soon as the ego vehicle completes the configured number of laps.",
    )
    parser.add_argument(
        "--no-terminate-on-finish",
        dest="terminate_on_finish",
        action="store_false",
        help="Keep running after crossing the finish line; success will still be reported in env info.",
    )
    parser.add_argument("--env-seed", type=int, default=100)
    parser.add_argument("--self-play-seed", type=int, default=1000)
    parser.add_argument(
        "--network-seed",
        type=int,
        default=42,
        help="Used only when --model-path is not provided.",
    )
    parser.add_argument("--model-path", default=None, help="Optional model checkpoint (.pth).")
    parser.add_argument("--n-simulations", type=int, default=5)
    parser.add_argument("--c-puct", type=float, default=2.5)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--temperature-drop-step", type=int, default=None)
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    parser.add_argument("--root-exploration-fraction", type=float, default=None)
    parser.add_argument(
        "--mcts-max-root-visits",
        type=int,
        default=None,
        help="Stop rollouts for current decision once root visit count reaches this cap.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for neural inference/training in workers: auto|cpu|cuda|cuda:0",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help=(
            "Maximum number of compatible CUDA devices to use when --device=auto or --device=cuda. "
            "Workers are assigned round-robin across the selected GPUs."
        ),
    )
    parser.add_argument(
        "--gpu-indices",
        default=None,
        help=(
            "Comma-separated CUDA device indices to use when --device=auto or --device=cuda, "
            "for example: 0,1,3"
        ),
    )
    parser.add_argument("--n-residual-layers", type=int, default=10)
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="Optional hard cap for self-play steps per episode. Defaults to no extra cap beyond the environment's own duration.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1,
        help="Print MCTS/inference stats every N self-play steps inside each worker.",
    )
    parser.add_argument(
        "--worker-heartbeat-interval",
        type=float,
        default=0.0,
        help=(
            "Print a liveness heartbeat from each worker every N seconds even if no step has completed yet. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--manager-heartbeat-interval",
        type=float,
        default=0.0,
        help=(
            "Print a liveness heartbeat from the parent process every N seconds while waiting for results. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--print-actions",
        action="store_true",
        help="Print per-step actions inside each worker (verbose, mostly for debugging).",
    )
    parser.add_argument(
        "--output-dir",
        default="AlphaZero-based-autonomous-driving/outputs/racetrack_self_play_parallel",
    )
    parser.add_argument(
        "--result-timeout",
        type=float,
        default=1800.0,
        help="Max seconds to wait for one worker result.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = SELF_PLAY_CONFIG
    temperature = config.temperature if args.temperature is None else args.temperature
    temperature_drop_step = (
        config.temperature_drop_step
        if args.temperature_drop_step is None
        else args.temperature_drop_step
    )
    root_dirichlet_alpha = (
        config.root_dirichlet_alpha
        if args.dirichlet_alpha is None
        else args.dirichlet_alpha
    )
    root_exploration_fraction = (
        config.root_exploration_fraction
        if args.root_exploration_fraction is None
        else args.root_exploration_fraction
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_devices = _resolve_worker_devices(args)
    unique_worker_devices = list(dict.fromkeys(worker_devices))

    model_path = None
    created_temp_model = False

    if args.model_path:
        model_path = Path(args.model_path).expanduser().resolve()
    else:
        torch.manual_seed(args.network_seed)
        network = AlphaZeroNetwork(
            input_shape=config.input_shape,
            n_residual_layers=args.n_residual_layers,
            n_actions=config.n_actions,
        )
        model_path = output_dir / "_initial_model_for_parallel_self_play.pth"
        torch.save(network.state_dict(), model_path)
        created_temp_model = True

    env_config = build_racetrack_env_config(
        duration=args.duration,
        other_vehicles=args.other_vehicles,
        finish_laps=args.finish_laps,
        terminate_on_finish=args.terminate_on_finish,
    )

    print(f"env_id={args.env_id}")
    print(f"workers={args.workers}, episodes_per_worker={args.episodes_per_worker}")
    print(f"n_actions={config.n_actions}, n_simulations={args.n_simulations}")
    print(f"finish_laps={args.finish_laps}, terminate_on_finish={args.terminate_on_finish}")
    print(f"temperature={temperature}, temperature_drop_step={temperature_drop_step}")
    print(f"dirichlet_alpha={root_dirichlet_alpha}, root_exploration_fraction={root_exploration_fraction}")
    print(f"mcts_max_root_visits={args.mcts_max_root_visits}")
    print(f"device={args.device}")
    print(f"resolved_worker_devices={unique_worker_devices}")
    print(f"progress_interval={args.progress_interval}")
    print(f"worker_heartbeat_interval={args.worker_heartbeat_interval}")
    print(f"manager_heartbeat_interval={args.manager_heartbeat_interval}")
    print(
        "worker_device_assignments="
        + ", ".join(
            f"worker_{worker_id}:{device}" for worker_id, device in enumerate(worker_devices)
        )
    )
    print(f"model_path={model_path}")
    print(f"output_dir={output_dir}")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for worker_id in range(args.workers):
        worker_env_seed = args.env_seed + worker_id * 10_000
        worker_self_play_seed = args.self_play_seed + worker_id * 10_000
        task = {
            "worker_id": worker_id,
            "episodes_per_worker": args.episodes_per_worker,
            "env_id": args.env_id,
            "env_config": env_config,
            "env_seed": worker_env_seed,
            "self_play_seed": worker_self_play_seed,
            "network_seed": args.network_seed,
            "model_path": str(model_path),
            "output_dir": str(output_dir),
            "input_shape": config.input_shape,
            "stack_config": config.stack,
            "n_actions": config.n_actions,
            "n_residual_layers": args.n_residual_layers,
            "n_simulations": args.n_simulations,
            "c_puct": args.c_puct,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "mcts_max_root_visits": args.mcts_max_root_visits,
            "device": worker_devices[worker_id],
            "temperature": temperature,
            "temperature_drop_step": temperature_drop_step,
            "add_root_dirichlet_noise": True,
            "root_dirichlet_alpha": root_dirichlet_alpha,
            "root_exploration_fraction": root_exploration_fraction,
            "torch_threads_per_worker": args.torch_threads_per_worker,
            "print_actions": args.print_actions,
            "max_steps_per_episode": args.max_steps_per_episode,
            "progress_interval": args.progress_interval,
            "worker_heartbeat_interval": args.worker_heartbeat_interval,
        }
        process = ctx.Process(target=_worker_entry, args=(task, result_queue))
        process.start()
        processes.append(process)
        print(
            f"[manager] started worker_{worker_id} pid={process.pid} device={worker_devices[worker_id]}",
            flush=True,
        )

    results = []
    pending_worker_ids = set(range(args.workers))
    wait_start = time.perf_counter()
    last_result_at = wait_start
    manager_heartbeat_interval = float(args.manager_heartbeat_interval)
    while pending_worker_ids:
        since_last_result = time.perf_counter() - last_result_at
        remaining_before_timeout = float(args.result_timeout) - since_last_result
        if remaining_before_timeout <= 0:
            pending_summary = ", ".join(f"worker_{worker_id}" for worker_id in sorted(pending_worker_ids))
            results.append(
                {
                    "ok": False,
                    "worker_id": None,
                    "error": (
                        "Timed out waiting for worker result. "
                        f"pending=[{pending_summary}] inactivity={_format_duration(since_last_result)}"
                    ),
                }
            )
            break

        queue_timeout = remaining_before_timeout
        if manager_heartbeat_interval > 0:
            queue_timeout = min(queue_timeout, manager_heartbeat_interval)
        try:
            item = result_queue.get(timeout=queue_timeout)
            results.append(item)
            result_worker_id = (
                item["result"]["worker_id"] if item.get("ok", False) else item.get("worker_id")
            )
            if result_worker_id is not None:
                pending_worker_ids.discard(int(result_worker_id))
            last_result_at = time.perf_counter()
            completed_workers = int(args.workers) - len(pending_worker_ids)
            print(
                f"[manager] received result from worker={result_worker_id} "
                f"ok={item.get('ok', False)} completed={completed_workers}/{args.workers}",
                flush=True,
            )
        except Empty:
            elapsed = time.perf_counter() - wait_start
            alive_workers = [
                f"worker_{worker_id}(pid={processes[worker_id].pid},device={worker_devices[worker_id]})"
                for worker_id in sorted(pending_worker_ids)
                if processes[worker_id].is_alive()
            ]
            stalled_workers = sorted(
                worker_id for worker_id in pending_worker_ids if not processes[worker_id].is_alive()
            )
            print(
                f"[manager] heartbeat elapsed={_format_duration(elapsed)} "
                f"completed={int(args.workers) - len(pending_worker_ids)}/{args.workers} "
                f"pending={sorted(pending_worker_ids)} "
                f"alive={alive_workers} dead_without_result={stalled_workers}",
                flush=True,
            )

    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    failures = [item for item in results if not item.get("ok", False)]
    if failures:
        print("parallel self-play failed:")
        for failure in failures:
            print(f"  worker={failure.get('worker_id')} error:\n{failure.get('error')}")
        raise SystemExit(1)

    worker_results = sorted((item["result"] for item in results), key=lambda x: x["worker_id"])
    total_samples = sum(item["total_samples"] for item in worker_results)
    total_episodes = sum(len(item["episode_summaries"]) for item in worker_results)

    print("parallel self-play completed.")
    print(f"total_episodes={total_episodes}")
    print(f"total_samples={total_samples}")
    for item in worker_results:
        print(f"worker={item['worker_id']} samples={item['total_samples']}")

    if created_temp_model:
        print(f"temporary_model={model_path}")


if __name__ == "__main__":
    main()

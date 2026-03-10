import argparse
import multiprocessing as mp
import os
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
    try:
        from AlphaZero.core.settings import SELF_PLAY_CONFIG
        from AlphaZero.network.alphazero_network import AlphaZeroNetwork
        from AlphaZero.training.trainer import AlphaZeroTrainer
    except ModuleNotFoundError as nested_exc:
        if nested_exc.name != "AlphaZero":
            raise
        from ..core.settings import SELF_PLAY_CONFIG
        from ..network.alphazero_network import AlphaZeroNetwork
        from ..training.trainer import AlphaZeroTrainer


def build_racetrack_env_config(duration: int, other_vehicles: int) -> dict:
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
        batch_size=int(task["batch_size"]),
        epochs=int(task["epochs"]),
        stack_config=task["stack_config"],
        n_actions=int(task["n_actions"]),
        verbose=False,
        device=task["device"],
        max_root_visits=task["mcts_max_root_visits"],
    )

    worker_id = int(task["worker_id"])
    episodes_per_worker = int(task["episodes_per_worker"])
    n_actions = int(task["n_actions"])
    print_actions = bool(task["print_actions"])
    progress_interval = int(task["progress_interval"])
    max_steps_per_episode = task["max_steps_per_episode"]
    max_steps_per_episode = (
        None if max_steps_per_episode is None else int(max_steps_per_episode)
    )
    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[worker {worker_id}] pid={os.getpid()} "
        f"episodes={episodes_per_worker} n_simulations={task['n_simulations']} "
        f"device={trainer.device}",
        flush=True,
    )

    episode_summaries = []
    total_samples = 0

    for episode_idx in range(episodes_per_worker):
        trainer.training_data.clear()
        trainer.action_list.clear()
        trainer.verbose = print_actions

        episode_seed = int(task["self_play_seed"]) + episode_idx
        print(
            f"[worker {worker_id}] episode {episode_idx + 1}/{episodes_per_worker} "
            f"start seed={episode_seed}",
            flush=True,
        )
        t0 = time.perf_counter()

        def _step_callback(info: dict):
            step = int(info["step"])
            done = bool(info["done"])
            if progress_interval > 0 and (step % progress_interval == 0 or done):
                print(
                    f"[worker {worker_id}] episode {episode_idx + 1}/{episodes_per_worker} "
                    f"step={step} action={info['action']} done={done}",
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
        print(
            f"[worker {worker_id}] episode {episode_idx + 1}/{episodes_per_worker} "
            f"done samples={sample_count} steps={len(trainer.action_list)} "
            f"time={elapsed:.2f}s saved={episode_path.name}",
            flush=True,
        )

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
    parser.add_argument("--n-residual-layers", type=int, default=10)
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=40,
        help="Hard cap for self-play steps per episode (safety against very long episodes).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=5,
        help="Print progress every N self-play steps inside each worker.",
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
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    )

    print(f"env_id={args.env_id}")
    print(f"workers={args.workers}, episodes_per_worker={args.episodes_per_worker}")
    print(f"n_actions={config.n_actions}, n_simulations={args.n_simulations}")
    print(f"mcts_max_root_visits={args.mcts_max_root_visits}")
    print(f"device={args.device}")
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
            "mcts_max_root_visits": args.mcts_max_root_visits,
            "device": args.device,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "torch_threads_per_worker": args.torch_threads_per_worker,
            "print_actions": args.print_actions,
            "max_steps_per_episode": args.max_steps_per_episode,
            "progress_interval": args.progress_interval,
        }
        process = ctx.Process(target=_worker_entry, args=(task, result_queue))
        process.start()
        processes.append(process)

    results = []
    for _ in range(args.workers):
        try:
            item = result_queue.get(timeout=args.result_timeout)
            results.append(item)
        except Empty:
            results.append({"ok": False, "worker_id": None, "error": "Timed out waiting for worker result."})

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

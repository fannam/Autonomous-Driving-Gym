import argparse
import os
import time

# Avoid ALSA warnings on headless/no-audio machines.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym

import highway_env  # noqa: F401


def build_detailed_observation_config(
    on_road_mode: str,
    on_road_soft_mode: bool,
    on_road_subsamples: int,
    presence_subsamples: int,
) -> dict:
    return {
        "type": "DetailedOccupancyGrid",
        "features": [
            "presence",
            "vx",
            "vy",
            "speed",
            "cos_h",
            "sin_h",
            "distance",
            "ttc",
            "on_lane",
            "on_road",
        ],
        "grid_size": [[-50, 50], [-12, 12]],
        "grid_step": [1.0, 1.0],
        "absolute": False,
        "align_to_vehicle_axes": True,
        "clip": True,
        "as_image": False,
        "include_ego_vehicle": True,
        "on_road_mode": on_road_mode,
        "on_road_soft_mode": on_road_soft_mode,
        "on_road_subsamples": on_road_subsamples,
        "presence_subsamples": presence_subsamples,
        "vehicle_footprint": True,
        "footprint_margin": 1.0,
        "ttc_horizon": 10.0,
        "distance_normalization": 120.0,
    }


def choose_action(env: gym.Env, policy: str) -> int:
    if policy == "idle":
        if hasattr(env.action_space, "n") and env.action_space.n >= 2:
            return 1
    return env.action_space.sample()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark highway-env step() throughput."
    )
    parser.add_argument("--env-id", default="highway-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000, help="Steps per episode.")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--policy",
        choices=["idle", "random"],
        default="idle",
        help="Action policy during benchmark.",
    )
    parser.add_argument(
        "--render-mode",
        choices=["none", "rgb_array", "human"],
        default="none",
        help="Render mode for env creation. Use 'none' for pure step benchmark.",
    )
    parser.add_argument(
        "--detailed-observation",
        action="store_true",
        help="Use DetailedOccupancyGrid instead of env default observation.",
    )
    parser.add_argument(
        "--on-road-mode",
        choices=["area", "centerline"],
        default="area",
        help="Only used with --detailed-observation.",
    )
    parser.add_argument(
        "--presence-subsamples",
        type=int,
        default=5,
        help="Only used with --detailed-observation.",
    )
    parser.add_argument(
        "--on-road-soft-mode",
        action="store_true",
        help="Only used with --detailed-observation: use fractional road occupancy per cell.",
    )
    parser.add_argument(
        "--on-road-subsamples",
        type=int,
        default=3,
        help="Only used with --detailed-observation and --on-road-soft-mode.",
    )
    return parser.parse_args()


def run_warmup(env: gym.Env, steps: int, seed: int, policy: str) -> None:
    if steps <= 0:
        return
    obs, _ = env.reset(seed=seed)
    _ = obs
    for _ in range(steps):
        action = choose_action(env, policy)
        _, _, done, truncated, _ = env.step(action)
        if done or truncated:
            env.reset()


def run_benchmark(
    env: gym.Env, episodes: int, steps_per_episode: int, seed: int, policy: str
) -> tuple[float, int]:
    total_step_time = 0.0
    total_steps = 0

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        step_count = 0
        while step_count < steps_per_episode:
            action = choose_action(env, policy)
            t0 = time.perf_counter()
            _, _, done, truncated, _ = env.step(action)
            total_step_time += time.perf_counter() - t0
            step_count += 1
            total_steps += 1
            if done or truncated:
                env.reset()

    return total_step_time, total_steps


def main() -> None:
    args = parse_args()
    render_mode = None if args.render_mode == "none" else args.render_mode
    env = gym.make(args.env_id, render_mode=render_mode)

    if args.detailed_observation:
        env.unwrapped.configure(
            {
                "observation": build_detailed_observation_config(
                    on_road_mode=args.on_road_mode,
                    on_road_soft_mode=args.on_road_soft_mode,
                    on_road_subsamples=args.on_road_subsamples,
                    presence_subsamples=args.presence_subsamples,
                )
            }
        )

    print(f"env={args.env_id}")
    print(f"render_mode={args.render_mode}")
    print(f"policy={args.policy}")
    print(f"episodes={args.episodes}, steps_per_episode={args.steps}")
    print(f"warmup_steps={args.warmup_steps}")
    if args.detailed_observation:
        print(
            "observation=DetailedOccupancyGrid "
            f"(on_road_mode={args.on_road_mode}, "
            f"on_road_soft_mode={args.on_road_soft_mode}, "
            f"on_road_subsamples={args.on_road_subsamples}, "
            f"presence_subsamples={args.presence_subsamples})"
        )
    else:
        print("observation=env_default")

    try:
        run_warmup(
            env=env,
            steps=args.warmup_steps,
            seed=args.seed,
            policy=args.policy,
        )
        t_bench_start = time.perf_counter()
        total_step_time, total_steps = run_benchmark(
            env=env,
            episodes=args.episodes,
            steps_per_episode=args.steps,
            seed=args.seed,
            policy=args.policy,
        )
        bench_wall = time.perf_counter() - t_bench_start
    finally:
        env.close()

    steps_per_sec_step_only = total_steps / total_step_time if total_step_time > 0 else 0.0
    ms_per_step_step_only = 1000.0 * total_step_time / total_steps if total_steps else 0.0
    steps_per_sec_wall = total_steps / bench_wall if bench_wall > 0 else 0.0

    print("----- benchmark result -----")
    print(f"total_steps={total_steps}")
    print(f"step_time_total={total_step_time:.6f}s (sum of each env.step call)")
    print(f"wall_time_total={bench_wall:.6f}s (includes loop/reset overhead)")
    print(f"step_only_throughput={steps_per_sec_step_only:.2f} steps/s")
    print(f"step_only_latency={ms_per_step_step_only:.3f} ms/step")
    print(f"wall_throughput={steps_per_sec_wall:.2f} steps/s")


if __name__ == "__main__":
    main()

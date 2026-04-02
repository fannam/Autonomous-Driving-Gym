import argparse
from typing import Iterable

import gymnasium as gym
import numpy as np

import highway_env  # noqa: F401
from highway_env.envs.common.observation import observation_factory


DEFAULT_FEATURES = [
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
]


DEFAULT_OCCUPANCY_FEATURES = ["presence", "vx", "vy", "on_road"]


def build_detailed_observation_config(as_image: bool) -> dict:
    return {
        "type": "DetailedOccupancyGrid",
        "features": DEFAULT_FEATURES,
        # 1m/cell: vehicle ~5x2 cells, lane width ~4 cells.
        # Keep window moderate to avoid overly large tensors.
        "grid_size": [[-50, 50], [-12, 12]],
        "grid_step": [1.0, 1.0],
        "absolute": False,
        "align_to_vehicle_axes": True,
        "clip": True,
        "as_image": as_image,
        "include_ego_vehicle": True,
        "on_road_mode": "area",
        "on_road_soft_mode": True,
        "on_road_subsamples": 3,
        "presence_subsamples": 5,
        "vehicle_footprint": True,
        "footprint_margin": 1.0,
        "ttc_horizon": 10.0,
        "distance_normalization": 120.0,
    }


def build_default_observation_config(as_image: bool) -> dict:
    return {
        "type": "OccupancyGrid",
        "features": DEFAULT_OCCUPANCY_FEATURES,
        "grid_size": [[-50, 50], [-12, 12]],
        "grid_step": [1.0, 1.0],
        "absolute": False,
        "align_to_vehicle_axes": True,
        "clip": True,
        "as_image": as_image,
    }


def summarize_channels(obs: np.ndarray, features: Iterable[str]) -> str:
    lines = []
    for idx, feature in enumerate(features):
        channel = obs[idx]
        non_zero_ratio = float(np.count_nonzero(channel) / channel.size)
        lines.append(
            f"  - {feature:<10s} min={channel.min(): .3f} "
            f"max={channel.max(): .3f} mean={channel.mean(): .3f} "
            f"non_zero={non_zero_ratio:.3f}"
        )
    return "\n".join(lines)


def _parse_channel_indices(print_channels: str, features: list[str]) -> list[int]:
    if print_channels.strip().lower() == "all":
        return list(range(len(features)))

    channel_indices: list[int] = []
    for token in print_channels.split(","):
        item = token.strip()
        if not item:
            continue
        if item.isdigit():
            index = int(item)
            if 0 <= index < len(features):
                channel_indices.append(index)
            else:
                print(f"warning: channel index out of range for current grid: {item}")
        else:
            if item not in features:
                print(
                    f"warning: channel '{item}' not found in current grid "
                    f"(available: {', '.join(features)})"
                )
                continue
            channel_indices.append(features.index(item))
    if not channel_indices:
        print("warning: no valid channel selected for current grid.")
    return channel_indices


def print_grid(
    title: str, obs: np.ndarray, features: list[str], print_channels: str
) -> None:
    channel_indices = _parse_channel_indices(print_channels, features)
    if not channel_indices:
        return
    np.set_printoptions(threshold=np.inf, linewidth=200, precision=3, suppress=True)
    print(f"{title}:")
    print(f"  full shape={obs.shape}")
    for idx in channel_indices:
        print(f"  channel[{idx}]={features[idx]} shape={obs[idx].shape}")
        print(obs[idx])


def print_observation_summary(
    label: str,
    obs: np.ndarray,
    features: list[str],
    print_grid_enabled: bool,
    print_channels: str,
) -> None:
    print(f"{label}: shape={obs.shape}, dtype={obs.dtype}")
    print(summarize_channels(obs, features))
    if print_grid_enabled:
        print_grid(title=f"{label} raw grid tensor", obs=obs, features=features, print_channels=print_channels)


def run_episode(
    env: gym.Env,
    max_steps: int,
    detailed_observer,
    default_observer,
    detailed_features: list[str],
    default_features: list[str],
    seed: int,
    print_grid_enabled: bool,
    print_channels: str,
) -> None:
    env.reset(seed=seed)
    detailed_obs = detailed_observer.observe()
    default_obs = default_observer.observe()
    print("reset:")
    print_observation_summary(
        label="  DetailedOccupancyGrid",
        obs=detailed_obs,
        features=detailed_features,
        print_grid_enabled=print_grid_enabled,
        print_channels=print_channels,
    )
    print_observation_summary(
        label="  OccupancyGrid (default)",
        obs=default_obs,
        features=default_features,
        print_grid_enabled=print_grid_enabled,
        print_channels=print_channels,
    )

    done = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not (done or truncated) and steps < max_steps:
        action = env.action_space.sample()
        _, reward, done, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1

        if steps % 10 == 0 or done or truncated or steps == max_steps:
            print(
                f"step={steps:03d} reward={reward:+.3f} total_reward={total_reward:+.3f} "
                f"done={done} truncated={truncated}"
            )

    detailed_obs = detailed_observer.observe()
    default_obs = default_observer.observe()
    print("episode end:")
    print_observation_summary(
        label="  DetailedOccupancyGrid",
        obs=detailed_obs,
        features=detailed_features,
        print_grid_enabled=print_grid_enabled,
        print_channels=print_channels,
    )
    print_observation_summary(
        label="  OccupancyGrid (default)",
        obs=default_obs,
        features=default_features,
        print_grid_enabled=print_grid_enabled,
        print_channels=print_channels,
    )
    print("-" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick smoke-test for DetailedOccupancyGrid observation."
    )
    parser.add_argument("--env-id", default="highway-v0", help="Gymnasium env id.")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes.")
    parser.add_argument(
        "--max-steps", type=int, default=60, help="Max steps per episode."
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with human mode.",
    )
    parser.add_argument(
        "--as-image",
        action="store_true",
        help="Use image encoding output [0, 255] instead of float channels.",
    )
    parser.add_argument(
        "--print-grid",
        action="store_true",
        help="Print raw observation grid tensor values.",
    )
    parser.add_argument(
        "--print-channels",
        default="all",
        help=(
            "Channels to print when --print-grid is set. "
            "Use 'all', names (e.g. presence,ttc), or indices (e.g. 0,7)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = "human" if args.render else None
    env = gym.make(args.env_id, render_mode=render_mode)
    detailed_observation_config = build_detailed_observation_config(as_image=args.as_image)
    default_observation_config = build_default_observation_config(as_image=args.as_image)

    # Keep env observation config as detailed; default occupancy is observed in parallel
    # from the same env state for direct comparison.
    env.unwrapped.configure({"observation": detailed_observation_config})
    detailed_observer = observation_factory(env.unwrapped, detailed_observation_config)
    default_observer = observation_factory(env.unwrapped, default_observation_config)

    print(f"env={args.env_id}")
    print(f"detailed observation type={detailed_observation_config['type']}")
    print(f"detailed features={detailed_observation_config['features']}")
    print(f"default observation type={default_observation_config['type']}")
    print(f"default features={default_observation_config['features']}")

    try:
        for episode_idx in range(args.episodes):
            print(f"episode {episode_idx + 1}/{args.episodes}")
            run_episode(
                env=env,
                max_steps=args.max_steps,
                detailed_observer=detailed_observer,
                default_observer=default_observer,
                detailed_features=detailed_observation_config["features"],
                default_features=default_observation_config["features"],
                seed=args.seed + episode_idx,
                print_grid_enabled=args.print_grid,
                print_channels=args.print_channels,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()

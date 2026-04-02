import argparse
import os
import time

# Avoid ALSA warnings on headless/no-audio machines.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import numpy as np

import highway_env  # noqa: F401
from highway_env.road.lane import AbstractLane


def build_observation_config(
    on_road_mode: str, on_road_soft_mode: bool, on_road_subsamples: int
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
        "presence_subsamples": 5,
        "vehicle_footprint": True,
        "footprint_margin": 1.0,
        "ttc_horizon": 10.0,
        "distance_normalization": 120.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize highway road and occupancy channels."
    )
    parser.add_argument("--env-id", default="highway-v0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--policy",
        choices=["idle", "random"],
        default="idle",
        help="idle: keep-lane action for stable visualization, random: random actions.",
    )
    parser.add_argument(
        "--on-road-mode",
        choices=["area", "centerline"],
        default="area",
        help="How on_road channel is rasterized in DetailedOccupancyGrid.",
    )
    parser.add_argument(
        "--on-road-soft-mode",
        action="store_true",
        help="Use fractional road occupancy ratio per cell in on_road channel.",
    )
    parser.add_argument(
        "--on-road-subsamples",
        type=int,
        default=3,
        help="Subsamples per axis for on_road soft occupancy.",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Use native pygame window (when --plot is disabled).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show matplotlib window with RGB frame + selected occupancy channel.",
    )
    parser.add_argument(
        "--plot-channel",
        choices=["on_road", "on_lane"],
        default="on_lane",
        help="Which occupancy channel to show in the matplotlib panel.",
    )
    parser.add_argument(
        "--print-on-road",
        action="store_true",
        help="Print on_road occupancy grid in terminal (at reset and episode end).",
    )
    parser.add_argument(
        "--print-on-lane",
        action="store_true",
        help="Print on_lane occupancy grid in terminal (at reset and episode end).",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=0,
        help="When >0 and print flag is enabled, also print every N steps.",
    )
    parser.add_argument(
        "--print-float",
        action="store_true",
        help="With print flags, print float values instead of binary 0/1.",
    )
    parser.add_argument(
        "--print-lane-limits",
        action="store_true",
        help="Print lane boundaries/limits around ego (world, ego-frame, and grid index).",
    )
    parser.add_argument(
        "--lane-limits-every",
        type=int,
        default=0,
        help="When >0 and --print-lane-limits, also print every N steps.",
    )
    return parser.parse_args()


def choose_action(env: gym.Env, policy: str) -> int:
    if policy == "idle":
        if hasattr(env.action_space, "n") and env.action_space.n >= 2:
            return 1
    return env.action_space.sample()


def print_channel_grid(
    channel: np.ndarray, channel_name: str, title: str, as_float: bool
) -> None:
    coverage = float(np.count_nonzero(channel) / channel.size)
    print(f"{title}: shape={channel.shape}, coverage={coverage:.3f}")
    np.set_printoptions(threshold=np.inf, linewidth=220, precision=3, suppress=True)
    if as_float:
        print(channel)
    else:
        print((channel > 0.5).astype(np.uint8))


def _world_to_ego(
    position: np.ndarray, ego_position: np.ndarray, ego_heading: float
) -> np.ndarray:
    rel = position - ego_position
    c, s = np.cos(ego_heading), np.sin(ego_heading)
    return np.array([[c, s], [-s, c]]) @ rel


def _rel_to_grid_index(
    rel_xy: np.ndarray, grid_size: list[list[float]], grid_step: list[float]
) -> tuple[int, int]:
    i = int(np.floor((rel_xy[0] - grid_size[0][0]) / grid_step[0]))
    j = int(np.floor((rel_xy[1] - grid_size[1][0]) / grid_step[1]))
    return i, j


def print_lane_limits(env: gym.Env, observation_config: dict, title: str) -> None:
    road = env.unwrapped.road
    ego = env.unwrapped.vehicle
    lanes = sorted(road.network.lanes_dict().items(), key=lambda item: item[0])
    grid_size = observation_config["grid_size"]
    grid_step = observation_config["grid_step"]
    align_to_vehicle_axes = bool(observation_config.get("align_to_vehicle_axes", False))
    grid_h = int((grid_size[0][1] - grid_size[0][0]) / grid_step[0])
    grid_w = int((grid_size[1][1] - grid_size[1][0]) / grid_step[1])

    print(title)
    print(
        f"  ego_pos=({ego.position[0]:.2f}, {ego.position[1]:.2f}) "
        f"ego_heading={ego.heading:.3f}rad ego_lane={ego.lane_index}"
    )
    print(
        f"  grid_x=[{grid_size[0][0]}, {grid_size[0][1]}], "
        f"grid_y=[{grid_size[1][0]}, {grid_size[1][1]}], "
        f"shape=({grid_h}, {grid_w})"
    )

    y_mins = []
    y_maxs = []

    for lane_index, lane in lanes:
        long, lat = lane.local_coordinates(ego.position)
        width = float(lane.width_at(long))
        half_w = 0.5 * width
        center_world = lane.position(long, 0.0)
        left_world = lane.position(long, +half_w)
        right_world = lane.position(long, -half_w)

        if align_to_vehicle_axes:
            center_rel = _world_to_ego(center_world, ego.position, ego.heading)
            left_rel = _world_to_ego(left_world, ego.position, ego.heading)
            right_rel = _world_to_ego(right_world, ego.position, ego.heading)
        else:
            center_rel = center_world - ego.position
            left_rel = left_world - ego.position
            right_rel = right_world - ego.position

        center_cell = _rel_to_grid_index(center_rel, grid_size, grid_step)
        left_cell = _rel_to_grid_index(left_rel, grid_size, grid_step)
        right_cell = _rel_to_grid_index(right_rel, grid_size, grid_step)
        y_low, y_high = sorted([float(left_rel[1]), float(right_rel[1])])
        y_mins.append(y_low)
        y_maxs.append(y_high)

        in_grid = 0 <= center_cell[0] < grid_h and 0 <= center_cell[1] < grid_w

        print(
            f"  lane={lane_index} width={width:.2f}m "
            f"ego_lat_offset={lat:+.2f}m "
            f"y_bounds_ego=[{y_low:+.2f}, {y_high:+.2f}]m "
            f"center_cell={center_cell} left_cell={left_cell} right_cell={right_cell} "
            f"in_grid={in_grid}"
        )

    if y_mins and y_maxs:
        print(
            f"  road lateral envelope near ego: "
            f"[{min(y_mins):+.2f}, {max(y_maxs):+.2f}]m"
        )


def maybe_print_channel(
    obs: np.ndarray,
    features: list[str],
    channel_name: str,
    enabled: bool,
    title: str,
    as_float: bool,
) -> None:
    if not enabled:
        return
    if channel_name not in features:
        print(f"warning: channel '{channel_name}' not found in features.")
        return
    idx = features.index(channel_name)
    print_channel_grid(obs[idx], channel_name, title, as_float)


def main() -> None:
    args = parse_args()
    render_mode = "rgb_array" if args.plot else ("human" if args.human else "rgb_array")
    env = gym.make(args.env_id, render_mode=render_mode)

    observation_config = build_observation_config(
        on_road_mode=args.on_road_mode,
        on_road_soft_mode=args.on_road_soft_mode,
        on_road_subsamples=args.on_road_subsamples,
    )
    env.unwrapped.configure({"observation": observation_config})
    features = observation_config["features"]

    print(f"env={args.env_id}")
    print(f"lanes_count={env.unwrapped.config.get('lanes_count')}")
    print(f"lane_width={AbstractLane.DEFAULT_WIDTH}m")
    print(
        f"observation={observation_config['type']} "
        f"grid_size={observation_config['grid_size']} "
        f"grid_step={observation_config['grid_step']} "
        f"on_road_mode={observation_config['on_road_mode']} "
        f"on_road_soft_mode={observation_config['on_road_soft_mode']} "
        f"on_road_subsamples={observation_config['on_road_subsamples']}"
    )

    plot_enabled = bool(args.plot)
    plot_channel_idx = features.index(args.plot_channel)
    fig = None
    ax_rgb = None
    ax_occ = None
    im_rgb = None
    im_occ = None

    if plot_enabled:
        import matplotlib.pyplot as plt

        plt.ion()
        fig, (ax_rgb, ax_occ) = plt.subplots(1, 2, figsize=(12, 5))
        ax_rgb.set_title("Road (RGB)")
        ax_rgb.axis("off")
        ax_occ.set_title(f"{args.plot_channel} channel")
        ax_occ.set_xlabel("y cells")
        ax_occ.set_ylabel("x cells")

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            print(f"episode {episode + 1}/{args.episodes} obs_shape={obs.shape}")
            maybe_print_channel(
                obs=obs,
                features=features,
                channel_name="on_road",
                enabled=args.print_on_road,
                title="on_road @ reset",
                as_float=args.print_float,
            )
            maybe_print_channel(
                obs=obs,
                features=features,
                channel_name="on_lane",
                enabled=args.print_on_lane,
                title="on_lane @ reset",
                as_float=args.print_float,
            )
            if args.print_lane_limits:
                print_lane_limits(
                    env=env,
                    observation_config=observation_config,
                    title="lane limits @ reset",
                )

            for step in range(args.steps):
                action = choose_action(env, args.policy)
                obs, reward, done, truncated, _ = env.step(action)
                frame = env.render()

                if args.print_every > 0 and (step + 1) % args.print_every == 0:
                    maybe_print_channel(
                        obs=obs,
                        features=features,
                        channel_name="on_road",
                        enabled=args.print_on_road,
                        title=f"on_road @ step {step + 1}",
                        as_float=args.print_float,
                    )
                    maybe_print_channel(
                        obs=obs,
                        features=features,
                        channel_name="on_lane",
                        enabled=args.print_on_lane,
                        title=f"on_lane @ step {step + 1}",
                        as_float=args.print_float,
                    )

                if (
                    args.print_lane_limits
                    and args.lane_limits_every > 0
                    and (step + 1) % args.lane_limits_every == 0
                ):
                    print_lane_limits(
                        env=env,
                        observation_config=observation_config,
                        title=f"lane limits @ step {step + 1}",
                    )

                if plot_enabled and frame is not None:
                    channel = obs[plot_channel_idx]
                    if im_rgb is None:
                        im_rgb = ax_rgb.imshow(frame)
                        im_occ = ax_occ.imshow(
                            channel, cmap="gray", vmin=0, vmax=1, origin="lower"
                        )
                    else:
                        im_rgb.set_data(frame)
                        im_occ.set_data(channel)
                    ax_rgb.set_title(
                        f"Road (step={step + 1}, reward={reward:+.3f}, done={done or truncated})"
                    )
                    fig.canvas.draw_idle()
                    import matplotlib.pyplot as plt

                    plt.pause(1.0 / max(args.fps, 1e-6))
                else:
                    time.sleep(1.0 / max(args.fps, 1e-6))

                if done or truncated:
                    break

            maybe_print_channel(
                obs=obs,
                features=features,
                channel_name="on_road",
                enabled=args.print_on_road,
                title="on_road @ episode end",
                as_float=args.print_float,
            )
            maybe_print_channel(
                obs=obs,
                features=features,
                channel_name="on_lane",
                enabled=args.print_on_lane,
                title="on_lane @ episode end",
                as_float=args.print_float,
            )
            if args.print_lane_limits:
                print_lane_limits(
                    env=env,
                    observation_config=observation_config,
                    title="lane limits @ episode end",
                )
    finally:
        env.close()
        if plot_enabled:
            import matplotlib.pyplot as plt

            plt.ioff()
            plt.show()


if __name__ == "__main__":
    main()

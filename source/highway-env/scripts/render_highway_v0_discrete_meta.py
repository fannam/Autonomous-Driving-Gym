from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.repo_layout import HIGHWAY_ENV_ROOT, prepend_sys_path  # noqa: E402


prepend_sys_path(HIGHWAY_ENV_ROOT)

# Avoid ALSA warnings on headless/no-audio machines.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym  # noqa: E402
import highway_env  # noqa: F401,E402


CONTROLLED_AGENT_COLORS = (
    (50, 200, 0),
    (220, 30, 160),
)
SCRIPTED_POLICY_TO_LABEL = {
    "idle": "IDLE",
    "left": "LANE_LEFT",
    "right": "LANE_RIGHT",
    "faster": "FASTER",
    "slower": "SLOWER",
}


class ObserverProxy:
    def __init__(self, position: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render highway-v0 with 2 controlled agents using "
            "MultiAgentAction + DiscreteMetaAction."
        )
    )
    parser.add_argument("--env-id", type=str, default="highway-v0")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--render-mode",
        choices=("rgb_array", "human"),
        default="human",
        help="Use rgb_array for headless capture, human for a pygame window.",
    )
    parser.add_argument(
        "--ego-policy",
        choices=("idle", "random", "left", "right", "faster", "slower"),
        default="idle",
    )
    parser.add_argument(
        "--npc-policy",
        choices=("idle", "random", "left", "right", "faster", "slower"),
        default="idle",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("midpoint", "first", "second", "auto"),
        default="midpoint",
        help="midpoint keeps both controlled vehicles in one view when possible.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Only used to slow down human rendering.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--vehicles-count",
        type=int,
        default=12,
        help="Total vehicle count in the environment, including controlled ones.",
    )
    parser.add_argument("--lanes-count", type=int, default=4)
    parser.add_argument("--policy-frequency", type=int, default=1)
    parser.add_argument("--simulation-frequency", type=int, default=None)
    parser.add_argument("--screen-width", type=int, default=900)
    parser.add_argument("--screen-height", type=int, default=220)
    parser.add_argument(
        "--target-speeds",
        type=float,
        nargs="+",
        default=(20.0, 25.0, 30.0),
        help="Target speeds passed to DiscreteMetaAction.",
    )
    parser.add_argument(
        "--camera-padding-m",
        type=float,
        default=18.0,
        help="Extra world-space margin added around the 2 controlled vehicles.",
    )
    parser.add_argument(
        "--min-scaling",
        type=float,
        default=1.5,
        help="Lower bound for dynamic zoom.",
    )
    parser.add_argument(
        "--fixed-scaling",
        type=float,
        default=None,
        help="Override dynamic zoom with a constant pixels-per-meter scale.",
    )
    parser.add_argument("--save-frames-dir", type=Path, default=None)
    return parser.parse_args()


def build_env_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "controlled_vehicles": 2,
        "vehicles_count": int(args.vehicles_count),
        "lanes_count": int(args.lanes_count),
        "duration": int(args.duration),
        "policy_frequency": int(args.policy_frequency),
        "screen_width": int(args.screen_width),
        "screen_height": int(args.screen_height),
        "centering_position": [0.5, 0.5],
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
                "target_speeds": [float(speed) for speed in args.target_speeds],
            },
        },
    }
    if args.simulation_frequency is not None:
        config["simulation_frequency"] = int(args.simulation_frequency)
    if args.render_mode == "rgb_array":
        config["offscreen_rendering"] = True
    return config


def get_controlled_vehicles(env) -> tuple[Any, Any]:
    controlled_vehicles = tuple(getattr(env.unwrapped, "controlled_vehicles", ()))
    if len(controlled_vehicles) < 2:
        raise RuntimeError(
            f"Expected at least 2 controlled vehicles, got {len(controlled_vehicles)}."
        )
    return controlled_vehicles[0], controlled_vehicles[1]


def get_agent_action_types(env) -> tuple[Any, Any]:
    action_type = getattr(env.unwrapped, "action_type", None)
    agents_action_types = getattr(action_type, "agents_action_types", None)
    if agents_action_types is None or len(agents_action_types) < 2:
        raise RuntimeError(
            "Expected MultiAgentAction with 2 per-agent action types."
        )
    return agents_action_types[0], agents_action_types[1]


def colorize_controlled_vehicles(env) -> None:
    ego_vehicle, npc_vehicle = get_controlled_vehicles(env)
    ego_vehicle.color = CONTROLLED_AGENT_COLORS[0]
    npc_vehicle.color = CONTROLLED_AGENT_COLORS[1]


def describe_observation(obs: Any) -> str:
    if isinstance(obs, tuple):
        parts = [f"agent_{index}({describe_observation(item)})" for index, item in enumerate(obs)]
        return "tuple(" + ", ".join(parts) + ")"
    if isinstance(obs, dict):
        pieces = []
        for key, value in obs.items():
            array = np.asarray(value)
            pieces.append(f"{key}:shape={array.shape},dtype={array.dtype}")
        return "dict(" + ", ".join(pieces) + ")"
    array = np.asarray(obs)
    return f"shape={array.shape}, dtype={array.dtype}"


def vehicle_summary(vehicle: Any) -> str:
    position = np.asarray(getattr(vehicle, "position", (0.0, 0.0)), dtype=np.float32)
    return (
        f"type={vehicle.__class__.__name__} "
        f"pos=({float(position[0]):.2f}, {float(position[1]):.2f}) "
        f"speed={float(getattr(vehicle, 'speed', 0.0)):.2f} "
        f"heading={float(getattr(vehicle, 'heading', 0.0)):.2f} "
        f"lane={getattr(vehicle, 'lane_index', None)} "
        f"crashed={bool(getattr(vehicle, 'crashed', False))} "
        f"on_road={bool(getattr(vehicle, 'on_road', True))}"
    )


def print_vehicle_summaries(env) -> None:
    ego_vehicle, npc_vehicle = get_controlled_vehicles(env)
    print(f"[agent 0] {vehicle_summary(ego_vehicle)}")
    print(f"[agent 1] {vehicle_summary(npc_vehicle)}")


def action_label(agent_action_type: Any, action_index: int) -> str:
    actions = getattr(agent_action_type, "actions", None)
    if isinstance(actions, dict):
        return str(actions.get(int(action_index), action_index))
    return str(action_index)


def print_action_catalog(env) -> None:
    for agent_index, agent_action_type in enumerate(get_agent_action_types(env)):
        actions = getattr(agent_action_type, "actions", None)
        if not isinstance(actions, dict):
            continue
        labels = ", ".join(
            f"{int(action_index)}={action_name}"
            for action_index, action_name in sorted(actions.items())
        )
        print(f"[agent {agent_index}] actions: {labels}")


def resolve_policy_action(
    agent_action_type: Any,
    policy_name: str,
    rng: np.random.Generator,
) -> tuple[int, bool]:
    available_actions = tuple(
        int(action) for action in agent_action_type.get_available_actions()
    )
    if policy_name == "random":
        return int(rng.choice(np.asarray(available_actions, dtype=np.int64))), False

    label = SCRIPTED_POLICY_TO_LABEL[policy_name]
    actions_indexes = getattr(agent_action_type, "actions_indexes", None)
    if not isinstance(actions_indexes, dict) or label not in actions_indexes:
        raise RuntimeError(f"Could not resolve {label!r} for {agent_action_type!r}.")

    requested_action = int(actions_indexes[label])
    if requested_action in available_actions:
        return requested_action, False

    idle_action = int(actions_indexes["IDLE"])
    return idle_action, True


def build_joint_action(
    env,
    *,
    ego_policy: str,
    npc_policy: str,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[bool, bool]]:
    ego_action_type, npc_action_type = get_agent_action_types(env)
    ego_action, ego_fallback = resolve_policy_action(ego_action_type, ego_policy, rng)
    npc_action, npc_fallback = resolve_policy_action(npc_action_type, npc_policy, rng)
    return (ego_action, npc_action), (ego_fallback, npc_fallback)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError(f"Expected HxWxC frame, got shape={array.shape!r}.")
    if array.dtype == np.uint8:
        return array
    return np.clip(array, 0, 255).astype(np.uint8)


def save_frame(frame: np.ndarray, output_dir: Path, frame_index: int) -> Path:
    if Image is None:
        raise RuntimeError(
            "Saving frames requires Pillow. Install it or omit --save-frames-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / f"frame_{frame_index:03d}.png"
    Image.fromarray(normalize_frame(frame)).save(frame_path)
    return frame_path


def compute_camera_position(env, camera_mode: str) -> np.ndarray | None:
    ego_vehicle, npc_vehicle = get_controlled_vehicles(env)
    if camera_mode == "first":
        return np.asarray(ego_vehicle.position, dtype=np.float32)
    if camera_mode == "second":
        return np.asarray(npc_vehicle.position, dtype=np.float32)
    if camera_mode == "midpoint":
        return 0.5 * (
            np.asarray(ego_vehicle.position, dtype=np.float32)
            + np.asarray(npc_vehicle.position, dtype=np.float32)
        )
    if camera_mode == "auto":
        return None
    raise ValueError(f"Unsupported camera mode: {camera_mode!r}.")


def compute_dynamic_scaling(
    env,
    *,
    viewer,
    camera_position: np.ndarray,
    camera_padding_m: float,
    min_scaling: float,
    max_scaling: float,
) -> float:
    ego_vehicle, npc_vehicle = get_controlled_vehicles(env)
    controlled_positions = np.stack(
        [
            np.asarray(ego_vehicle.position, dtype=np.float32),
            np.asarray(npc_vehicle.position, dtype=np.float32),
        ],
        axis=0,
    )
    relative_positions = np.abs(controlled_positions - camera_position.reshape(1, 2))
    max_dx = float(np.max(relative_positions[:, 0]))
    max_dy = float(np.max(relative_positions[:, 1]))

    padding_x = float(camera_padding_m)
    padding_y = max(8.0, 0.5 * float(camera_padding_m))
    required_half_width = max(max_dx + padding_x, 1e-6)
    required_half_height = max(max_dy + padding_y, 1e-6)

    fit_x = 0.5 * float(viewer.sim_surface.get_width()) / required_half_width
    fit_y = 0.5 * float(viewer.sim_surface.get_height()) / required_half_height
    fit_scaling = min(fit_x, fit_y)
    return float(np.clip(fit_scaling, min_scaling, max_scaling))


def configure_viewer(
    env,
    *,
    camera_mode: str,
    fixed_scaling: float | None,
    camera_padding_m: float,
    min_scaling: float,
) -> None:
    viewer = getattr(env.unwrapped, "viewer", None)
    if viewer is None:
        return

    viewer.sim_surface.centering_position = [0.5, 0.5]
    camera_position = compute_camera_position(env, camera_mode)

    max_scaling = float(env.unwrapped.config.get("scaling", viewer.sim_surface.scaling))
    if fixed_scaling is not None:
        viewer.sim_surface.scaling = float(fixed_scaling)
    elif camera_position is not None:
        viewer.sim_surface.scaling = compute_dynamic_scaling(
            env,
            viewer=viewer,
            camera_position=camera_position,
            camera_padding_m=camera_padding_m,
            min_scaling=min_scaling,
            max_scaling=max_scaling,
        )

    if camera_position is None:
        viewer.observer_vehicle = None
        return

    observer_proxy = getattr(viewer, "_two_agent_observer_proxy", None)
    if observer_proxy is None:
        observer_proxy = ObserverProxy(camera_position)
        viewer._two_agent_observer_proxy = observer_proxy
    observer_proxy.position = np.asarray(camera_position, dtype=np.float32)
    viewer.observer_vehicle = observer_proxy


def describe_camera(env) -> str:
    viewer = getattr(env.unwrapped, "viewer", None)
    if viewer is None:
        return "viewer=None"

    observer_vehicle = getattr(viewer, "observer_vehicle", None)
    if observer_vehicle is None:
        center = None
    else:
        center = np.asarray(getattr(observer_vehicle, "position", None), dtype=np.float32)

    scaling = float(getattr(viewer.sim_surface, "scaling", 0.0))
    if center is None:
        return f"center=auto scaling={scaling:.3f}"
    return (
        f"center=({float(center[0]):.2f}, {float(center[1]):.2f}) "
        f"scaling={scaling:.3f}"
    )


def render_frame(
    env,
    *,
    output_dir: Path | None,
    frame_index: int,
    camera_mode: str,
    fixed_scaling: float | None,
    camera_padding_m: float,
    min_scaling: float,
) -> np.ndarray | None:
    env.render()
    configure_viewer(
        env,
        camera_mode=camera_mode,
        fixed_scaling=fixed_scaling,
        camera_padding_m=camera_padding_m,
        min_scaling=min_scaling,
    )
    frame = env.render()
    if frame is None:
        return None
    frame_array = np.asarray(frame)
    if output_dir is not None:
        frame_path = save_frame(frame_array, output_dir, frame_index)
        print(f"[frame {frame_index}] saved={frame_path}")
    return frame_array


def maybe_sleep(render_mode: str, fps: float) -> None:
    if render_mode != "human":
        return
    if fps <= 0:
        return
    time.sleep(1.0 / fps)


def print_step_info(
    env,
    *,
    step_index: int,
    joint_action: tuple[int, int],
    reward: Any,
    terminated: bool,
    truncated: bool,
    fallback_used: tuple[bool, bool],
) -> None:
    ego_action_type, npc_action_type = get_agent_action_types(env)
    ego_label = action_label(ego_action_type, joint_action[0])
    npc_label = action_label(npc_action_type, joint_action[1])
    fallback_suffix = ""
    if fallback_used[0] or fallback_used[1]:
        fallback_suffix = (
            f" fallback=(agent0={fallback_used[0]}, agent1={fallback_used[1]})"
        )
    print(
        f"[step {step_index}] "
        f"joint_action=({joint_action[0]}:{ego_label}, {joint_action[1]}:{npc_label}) "
        f"reward={reward} "
        f"terminated={terminated} "
        f"truncated={truncated}{fallback_suffix}"
    )


def main() -> int:
    args = parse_args()
    if args.save_frames_dir is not None and args.render_mode != "rgb_array":
        raise ValueError("--save-frames-dir only works with --render-mode rgb_array.")

    if args.render_mode == "human" and not os.environ.get("DISPLAY"):
        print(
            "[warn] DISPLAY is not set. Human rendering may fail on a headless machine. "
            "Use --render-mode rgb_array if needed."
        )

    env_config = build_env_config(args)
    env = gym.make(
        args.env_id,
        render_mode=args.render_mode,
        config=env_config,
    )
    rng = np.random.default_rng(args.seed)
    frame_index = 0

    print(
        "[render-highway-discrete-meta] "
        f"env_id={args.env_id} "
        f"render_mode={args.render_mode} "
        f"ego_policy={args.ego_policy} "
        f"npc_policy={args.npc_policy} "
        f"camera_mode={args.camera_mode} "
        f"seed={args.seed} "
        f"steps={args.steps}"
    )
    print(f"[config] {env_config}")

    try:
        observation, info = env.reset(seed=args.seed)
        colorize_controlled_vehicles(env)
        print(f"[reset] obs={describe_observation(observation)}")
        if info:
            print(f"[reset] info_keys={sorted(info.keys())}")
        print_action_catalog(env)
        print_vehicle_summaries(env)

        frame = render_frame(
            env,
            output_dir=args.save_frames_dir,
            frame_index=frame_index,
            camera_mode=args.camera_mode,
            fixed_scaling=args.fixed_scaling,
            camera_padding_m=args.camera_padding_m,
            min_scaling=args.min_scaling,
        )
        print(f"[camera] {describe_camera(env)}")
        if frame is not None:
            print(f"[frame {frame_index}] shape={tuple(frame.shape)} dtype={frame.dtype}")
            frame_index += 1
        maybe_sleep(args.render_mode, args.fps)

        for step_index in range(1, int(args.steps) + 1):
            joint_action, fallback_used = build_joint_action(
                env,
                ego_policy=args.ego_policy,
                npc_policy=args.npc_policy,
                rng=rng,
            )
            observation, reward, terminated, truncated, info = env.step(joint_action)

            print_step_info(
                env,
                step_index=step_index,
                joint_action=joint_action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                fallback_used=fallback_used,
            )
            print(f"[step {step_index}] obs={describe_observation(observation)}")
            print_vehicle_summaries(env)

            frame = render_frame(
                env,
                output_dir=args.save_frames_dir,
                frame_index=frame_index,
                camera_mode=args.camera_mode,
                fixed_scaling=args.fixed_scaling,
                camera_padding_m=args.camera_padding_m,
                min_scaling=args.min_scaling,
            )
            print(f"[camera] {describe_camera(env)}")
            if frame is not None:
                print(
                    f"[frame {frame_index}] shape={tuple(frame.shape)} dtype={frame.dtype}"
                )
                frame_index += 1

            maybe_sleep(args.render_mode, args.fps)

            if terminated or truncated:
                print(f"[stop] ended at step={step_index}")
                break
    finally:
        env.close()

    print(f"[summary] frames={frame_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

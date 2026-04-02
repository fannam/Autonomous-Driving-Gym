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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test rendering for highway-v0."
    )
    parser.add_argument("--env-id", type=str, default="highway-v0")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--render-mode",
        choices=("rgb_array", "human"),
        default="human",
        help="Use rgb_array for headless testing, human for a pygame window.",
    )
    parser.add_argument(
        "--policy",
        choices=("idle", "random"),
        default="idle",
        help="idle keeps the default lane on discrete action spaces.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Only used to slow down human rendering.",
    )
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--vehicles-count", type=int, default=None)
    parser.add_argument("--screen-width", type=int, default=None)
    parser.add_argument("--screen-height", type=int, default=None)
    parser.add_argument("--save-frames-dir", type=Path, default=None)
    return parser.parse_args()


def build_env_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if args.duration is not None:
        config["duration"] = int(args.duration)
    if args.vehicles_count is not None:
        config["vehicles_count"] = int(args.vehicles_count)
    if args.screen_width is not None:
        config["screen_width"] = int(args.screen_width)
    if args.screen_height is not None:
        config["screen_height"] = int(args.screen_height)
    if args.render_mode == "rgb_array":
        config["offscreen_rendering"] = True
    return config


def choose_action(env: gym.Env, policy: str) -> Any:
    if policy == "idle":
        if hasattr(env.action_space, "n") and env.action_space.n >= 2:
            return 1
        raise RuntimeError(
            "The idle policy only supports discrete action spaces with IDLE=1."
        )
    return env.action_space.sample()


def describe_observation(obs: Any) -> str:
    if isinstance(obs, dict):
        pieces = []
        for key, value in obs.items():
            array = np.asarray(value)
            pieces.append(f"{key}:shape={array.shape},dtype={array.dtype}")
        return "dict(" + ", ".join(pieces) + ")"
    array = np.asarray(obs)
    return f"shape={array.shape}, dtype={array.dtype}"


def vehicle_summary(env: gym.Env) -> str:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    if vehicle is None:
        return "vehicle=None"

    position = np.asarray(getattr(vehicle, "position", (0.0, 0.0)), dtype=np.float32)
    lane_index = getattr(vehicle, "lane_index", None)
    return (
        f"type={vehicle.__class__.__name__} "
        f"pos=({float(position[0]):.2f}, {float(position[1]):.2f}) "
        f"speed={float(getattr(vehicle, 'speed', 0.0)):.2f} "
        f"heading={float(getattr(vehicle, 'heading', 0.0)):.2f} "
        f"lane={lane_index} "
        f"crashed={bool(getattr(vehicle, 'crashed', False))} "
        f"on_road={bool(getattr(vehicle, 'on_road', True))}"
    )


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


def capture_frame(
    env: gym.Env, *, output_dir: Path | None, frame_index: int
) -> np.ndarray | None:
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
        config=env_config or None,
    )

    total_reward = 0.0
    frame_index = 0

    print(
        "[render-highway] "
        f"env_id={args.env_id} "
        f"render_mode={args.render_mode} "
        f"policy={args.policy} "
        f"seed={args.seed} "
        f"steps={args.steps}"
    )
    if env_config:
        print(f"[config] {env_config}")

    try:
        obs, info = env.reset(seed=args.seed)
        print(f"[reset] obs={describe_observation(obs)}")
        if info:
            print(f"[reset] info_keys={sorted(info.keys())}")
        print(f"[ego] {vehicle_summary(env)}")

        frame = capture_frame(
            env,
            output_dir=args.save_frames_dir,
            frame_index=frame_index,
        )
        if frame is not None:
            print(f"[frame {frame_index}] shape={tuple(frame.shape)} dtype={frame.dtype}")
            frame_index += 1
        maybe_sleep(args.render_mode, args.fps)

        for step_index in range(1, int(args.steps) + 1):
            action = choose_action(env, args.policy)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            print(
                f"[step {step_index}] "
                f"action={action} "
                f"reward={float(reward):.4f} "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )
            print(f"[step {step_index}] obs={describe_observation(obs)}")
            print(f"[ego] {vehicle_summary(env)}")

            speed = info.get("speed") if isinstance(info, dict) else None
            if speed is not None:
                print(f"[step {step_index}] speed={float(np.asarray(speed).item()):.4f}")

            frame = capture_frame(
                env,
                output_dir=args.save_frames_dir,
                frame_index=frame_index,
            )
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

    print(
        "[summary] "
        f"frames={frame_index} "
        f"total_reward={total_reward:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

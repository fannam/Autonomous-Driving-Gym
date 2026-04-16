from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

try:
    from AlphaZeroAdversarial.core.game import (
        get_available_actions,
        get_controlled_vehicles,
        get_scripted_action,
    )
    from AlphaZeroAdversarial.core.runtime_config import get_scenario_config_path
    from AlphaZeroAdversarial.environment.config import build_env_spec, init_env
except ModuleNotFoundError:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))
    from AlphaZeroAdversarial.core.game import (
        get_available_actions,
        get_controlled_vehicles,
        get_scripted_action,
    )
    from AlphaZeroAdversarial.core.runtime_config import get_scenario_config_path
    from AlphaZeroAdversarial.environment.config import build_env_spec, init_env


DEFAULT_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HIGHWAY_CONFIG_PATH = (
    DEFAULT_PACKAGE_ROOT / "configs" / "highway_adversarial.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a few steps of the adversarial highway-env setup and "
            "verify that exactly two controlled vehicles exist."
        )
    )
    parser.add_argument("--stage", type=str, default="self_play")
    parser.add_argument("--scenario-name", type=str, default="highway_adversarial")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_HIGHWAY_CONFIG_PATH)
    parser.add_argument("--env-id", type=str, default="highway-v0")
    parser.add_argument("--env-seed", type=int, default=21)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--controlled-vehicles", type=int, default=2)
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--ego-policy", choices=("idle", "random"), default="random")
    parser.add_argument("--npc-policy", choices=("idle", "random"), default="random")
    parser.add_argument("--sleep-s", type=float, default=0.15)
    parser.add_argument("--save-frames-dir", type=Path, default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--other-vehicles", type=int, default=None)
    parser.add_argument("--policy-frequency", type=int, default=None)
    parser.add_argument("--simulation-frequency", type=int, default=None)
    return parser.parse_args()


def _get_runtime_config_path(args: argparse.Namespace) -> Path | None:
    if args.config_path is not None:
        return Path(args.config_path).expanduser().resolve()
    if args.scenario_name:
        return get_scenario_config_path(str(args.scenario_name))
    return None


def _build_env_overrides(args: argparse.Namespace) -> dict:
    overrides = {
        "controlled_vehicles": int(args.controlled_vehicles),
    }
    if args.duration is not None:
        overrides["duration"] = int(args.duration)
    if args.other_vehicles is not None:
        overrides["other_vehicles"] = int(args.other_vehicles)
    if args.policy_frequency is not None:
        overrides["policy_frequency"] = int(args.policy_frequency)
    if args.simulation_frequency is not None:
        overrides["simulation_frequency"] = int(args.simulation_frequency)
    return overrides


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError(f"Expected HxWxC frame, got shape={array.shape!r}.")
    if array.dtype == np.uint8:
        return array
    return np.clip(array, 0, 255).astype(np.uint8)


def _save_frame(frame: np.ndarray, output_dir: Path, frame_index: int) -> Path:
    if Image is None:
        raise RuntimeError(
            "Saving frames requires Pillow. Install it or omit --save-frames-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / f"frame_{frame_index:03d}.png"
    Image.fromarray(_normalize_frame(frame)).save(frame_path)
    return frame_path


def _vehicle_summary(vehicle) -> str:
    position = np.asarray(getattr(vehicle, "position", (0.0, 0.0)), dtype=np.float32)
    return (
        f"type={vehicle.__class__.__name__} "
        f"pos=({float(position[0]):.2f}, {float(position[1]):.2f}) "
        f"speed={float(getattr(vehicle, 'speed', 0.0)):.2f} "
        f"heading={float(getattr(vehicle, 'heading', 0.0)):.2f} "
        f"on_road={bool(getattr(vehicle, 'on_road', True))} "
        f"crashed={bool(getattr(vehicle, 'crashed', False))}"
    )


def _select_agent_action(
    *,
    env,
    agent_index: int,
    policy_name: str,
    available_actions: tuple[int, ...],
    rng: np.random.Generator,
) -> int:
    if policy_name == "idle":
        action = get_scripted_action(env, agent_index, "idle")
        if action is None:
            raise RuntimeError(
                f"Could not determine idle action for agent_index={agent_index}."
            )
        return int(action)
    if not available_actions:
        raise RuntimeError(f"No available actions for agent_index={agent_index}.")
    return int(rng.choice(np.asarray(available_actions, dtype=np.int64)))


def _capture_frame(env, *, output_dir: Path | None, frame_index: int) -> np.ndarray | None:
    frame = env.render()
    if frame is None:
        return None
    if output_dir is not None:
        frame_path = _save_frame(frame, output_dir, frame_index)
        print(f"[frame] saved={frame_path}")
    return np.asarray(frame)


def main() -> int:
    args = parse_args()
    if int(args.controlled_vehicles) != 2:
        raise ValueError("This adversarial renderer expects exactly 2 controlled vehicles.")

    env_overrides = _build_env_overrides(args)
    config_path = _get_runtime_config_path(args)
    spec = build_env_spec(
        stage=args.stage,
        scenario_name=args.scenario_name,
        config_path=config_path,
        env_name=args.env_id,
        render_mode=args.render_mode,
        env_config_overrides=env_overrides,
    )
    print(
        "[render-steps] "
        f"scenario={spec.scenario_name} "
        f"config_path={config_path} "
        f"env_id={spec.env_id} "
        f"render_mode={spec.render_mode} "
        f"controlled_vehicles={spec.config.get('controlled_vehicles')}"
    )

    env = init_env(
        env_name=args.env_id,
        seed=args.env_seed,
        render_mode=args.render_mode,
        stage=args.stage,
        scenario_name=args.scenario_name,
        config_path=config_path,
        env_config_overrides=env_overrides,
    )
    rng = np.random.default_rng(args.env_seed)

    try:
        env.reset(seed=args.env_seed)
        controlled_vehicles = tuple(getattr(env.unwrapped, "controlled_vehicles", ()))
        if len(controlled_vehicles) != 2:
            raise RuntimeError(
                "Expected exactly 2 controlled vehicles, "
                f"found {len(controlled_vehicles)}."
            )

        action_space = getattr(env, "action_space", None)
        spaces = getattr(action_space, "spaces", None)
        if spaces is not None and len(spaces) != 2:
            raise RuntimeError(
                f"Expected 2 agent action spaces, found {len(spaces)}."
            )

        ego_vehicle, npc_vehicle = get_controlled_vehicles(env)
        print(f"[vehicle 0] {_vehicle_summary(ego_vehicle)}")
        print(f"[vehicle 1] {_vehicle_summary(npc_vehicle)}")

        frame_count = 0
        frame = _capture_frame(
            env,
            output_dir=args.save_frames_dir,
            frame_index=frame_count,
        )
        if frame is not None:
            print(f"[frame] index={frame_count} shape={tuple(frame.shape)}")
            frame_count += 1

        for step_index in range(int(args.steps)):
            ego_actions, npc_actions = get_available_actions(env)
            ego_action = _select_agent_action(
                env=env,
                agent_index=0,
                policy_name=args.ego_policy,
                available_actions=ego_actions,
                rng=rng,
            )
            npc_action = _select_agent_action(
                env=env,
                agent_index=1,
                policy_name=args.npc_policy,
                available_actions=npc_actions,
                rng=rng,
            )
            _, reward, terminated, truncated, info = env.step((ego_action, npc_action))

            ego_vehicle, npc_vehicle = get_controlled_vehicles(env)
            print(
                f"[step {step_index + 1}] "
                f"joint_action=({ego_action}, {npc_action}) "
                f"reward={reward} "
                f"terminated={bool(terminated)} "
                f"truncated={bool(truncated)}"
            )
            print(f"[step {step_index + 1}] vehicle_0 {_vehicle_summary(ego_vehicle)}")
            print(f"[step {step_index + 1}] vehicle_1 {_vehicle_summary(npc_vehicle)}")
            if info:
                print(f"[step {step_index + 1}] info_keys={sorted(info.keys())}")

            frame = _capture_frame(
                env,
                output_dir=args.save_frames_dir,
                frame_index=frame_count,
            )
            if frame is not None:
                print(f"[frame] index={frame_count} shape={tuple(frame.shape)}")
                frame_count += 1

            if args.render_mode == "human" and float(args.sleep_s) > 0.0:
                time.sleep(float(args.sleep_s))

            if bool(terminated) or bool(truncated):
                print(
                    f"[render-steps] stopping early after step={step_index + 1} "
                    f"terminated={bool(terminated)} truncated={bool(truncated)}"
                )
                break
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

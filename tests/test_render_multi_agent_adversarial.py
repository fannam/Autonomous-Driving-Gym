from __future__ import annotations

import argparse
import copy
import json
from typing import Any

import gymnasium as gym
import numpy as np

import highway_env  # noqa: F401
from tools.repo_layout import ALPHAZERO_ADVERSARIAL_ROOT

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


DEFAULT_CONFIG_PATH = (
    ALPHAZERO_ADVERSARIAL_ROOT / "configs" / "racetrack_adversarial.yaml"
)
CONTROLLED_AGENT_COLORS = (
    (50, 200, 0),
    (200, 0, 150),
)


class ObserverProxy:
    def __init__(self, position: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=np.float32)


def load_config(config_path: Path) -> dict[str, Any]:
    raw_text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        loaded = yaml.safe_load(raw_text)
    else:
        loaded = json.loads(raw_text)

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected top-level mapping in {config_path}, got {type(loaded).__name__}."
        )
    return loaded


def get_environment_payload(config: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("Config must define an `environment` mapping.")

    env_id = environment.get("env_id")
    env_config = environment.get("config")
    render_mode = environment.get("render_mode")

    if not isinstance(env_id, str) or not env_id:
        raise ValueError("Environment config must define a non-empty `env_id`.")
    if not isinstance(env_config, dict):
        raise ValueError("Environment config must define a `config` mapping.")

    return env_id, render_mode, env_config


def neutral_action_index(agent_action_type) -> int:
    actions_indexes = getattr(agent_action_type, "actions_indexes", None)
    if isinstance(actions_indexes, dict) and "IDLE" in actions_indexes:
        return int(actions_indexes["IDLE"])

    actions_per_axis = int(getattr(agent_action_type, "actions_per_axis", 0))
    size = int(getattr(agent_action_type, "size", 0))
    if actions_per_axis > 0 and size > 0:
        center = actions_per_axis // 2
        action_index = 0
        for _ in range(size):
            action_index = action_index * actions_per_axis + center
        return int(action_index)

    agent_space = agent_action_type.space()
    if not hasattr(agent_space, "n"):
        raise ValueError("Expected a discrete per-agent action space.")
    return int(agent_space.n // 2)


def scripted_joint_action(env, mode: str) -> tuple[int, int]:
    action_type = getattr(env.unwrapped, "action_type", None)
    agents_action_types = getattr(action_type, "agents_action_types", None)
    if agents_action_types is None or len(agents_action_types) < 2:
        raise RuntimeError("Expected a MultiAgentAction environment with two controlled agents.")

    if mode == "idle":
        neutral_actions = tuple(
            neutral_action_index(agent_action_type)
            for agent_action_type in agents_action_types[:2]
        )
        return int(neutral_actions[0]), int(neutral_actions[1])

    if mode == "random":
        return tuple(
            int(agent_space.sample())
            for agent_space in env.action_space.spaces[:2]
        )

    raise ValueError(f"Unsupported action mode: {mode!r}")


def ensure_multi_agent_contract(env, observation) -> None:
    controlled_vehicles = getattr(env.unwrapped, "controlled_vehicles", ())
    if len(controlled_vehicles) != 2:
        raise AssertionError(
            f"Expected exactly 2 controlled vehicles, got {len(controlled_vehicles)}."
        )

    if not isinstance(observation, tuple) or len(observation) != 2:
        raise AssertionError(
            f"Expected a tuple of 2 observations, got {type(observation).__name__}."
        )

    action_space = getattr(env, "action_space", None)
    if action_space is None or not hasattr(action_space, "spaces") or len(action_space.spaces) != 2:
        raise AssertionError("Expected a tuple action space for 2 agents.")


def get_controlled_vehicles(env) -> tuple[Any, Any]:
    controlled_vehicles = getattr(env.unwrapped, "controlled_vehicles", ())
    if len(controlled_vehicles) < 2:
        raise AssertionError(
            f"Expected at least 2 controlled vehicles, got {len(controlled_vehicles)}."
        )
    return controlled_vehicles[0], controlled_vehicles[1]


def colorize_controlled_vehicles(env) -> None:
    ego_vehicle, npc_vehicle = get_controlled_vehicles(env)
    ego_vehicle.color = CONTROLLED_AGENT_COLORS[0]
    npc_vehicle.color = CONTROLLED_AGENT_COLORS[1]


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
    raise ValueError(f"Unsupported camera mode: {camera_mode!r}")


def configure_viewer(
    env,
    *,
    camera_mode: str,
    scaling: float | None,
) -> None:
    viewer = getattr(env.unwrapped, "viewer", None)
    if viewer is None:
        return

    if scaling is not None:
        viewer.sim_surface.scaling = float(scaling)

    camera_position = compute_camera_position(env, camera_mode)
    if camera_position is None:
        viewer.observer_vehicle = None
        return

    observer_proxy = getattr(viewer, "_multi_agent_observer_proxy", None)
    if observer_proxy is None:
        observer_proxy = ObserverProxy(camera_position)
        viewer._multi_agent_observer_proxy = observer_proxy
    observer_proxy.position = np.asarray(camera_position, dtype=np.float32)
    viewer.observer_vehicle = observer_proxy


def inspect_render_result(
    frame: Any,
    step_label: str,
    render_mode: str | None,
) -> dict[str, Any]:
    if render_mode == "human":
        if frame is None:
            return {
                "result_type": "none",
                "frame_shape": None,
            }
        if isinstance(frame, np.ndarray):
            if frame.ndim != 3:
                raise AssertionError(
                    f"{step_label}: expected frame shape (H, W, C), got {frame.shape}."
                )
            if frame.shape[2] not in (3, 4):
                raise AssertionError(
                    f"{step_label}: expected 3 or 4 channels, got {frame.shape[2]}."
                )
            return {
                "result_type": "ndarray",
                "frame_shape": (
                    int(frame.shape[0]),
                    int(frame.shape[1]),
                    int(frame.shape[2]),
                ),
            }
        raise AssertionError(
            f"{step_label}: render_mode='human' expected None or numpy.ndarray, "
            f"got {type(frame).__name__}."
        )

    if not isinstance(frame, np.ndarray):
        raise AssertionError(
            f"{step_label}: render_mode={render_mode!r} must return a numpy array, "
            f"got {type(frame).__name__}."
        )
    if frame.ndim != 3:
        raise AssertionError(
            f"{step_label}: expected frame shape (H, W, C), got {frame.shape}."
        )
    if frame.shape[2] not in (3, 4):
        raise AssertionError(
            f"{step_label}: expected 3 or 4 channels, got {frame.shape[2]}."
        )
    return {
        "result_type": "ndarray",
        "frame_shape": (
            int(frame.shape[0]),
            int(frame.shape[1]),
            int(frame.shape[2]),
        ),
    }


def run_render_smoke_test(
    *,
    config_path: Path,
    steps: int,
    seed: int,
    render_mode: str | None,
    action_mode: str,
    camera_mode: str,
    scaling: float | None,
) -> dict[str, Any]:
    config = load_config(config_path)
    env_id, config_render_mode, env_config = get_environment_payload(config)
    resolved_render_mode = config_render_mode if render_mode is None else render_mode

    env = gym.make(
        env_id,
        config=copy.deepcopy(env_config),
        render_mode=resolved_render_mode,
    )

    try:
        observation, info = env.reset(seed=seed)
        ensure_multi_agent_contract(env, observation)
        colorize_controlled_vehicles(env)

        frame = env.render()
        configure_viewer(
            env,
            camera_mode=camera_mode,
            scaling=scaling,
        )
        frame = env.render()
        initial_render = inspect_render_result(frame, "reset", resolved_render_mode)

        rollout_steps = 0
        terminated = False
        truncated = False
        step_records = []

        for step_index in range(steps):
            joint_action = scripted_joint_action(env, action_mode)
            observation, reward, terminated, truncated, info = env.step(joint_action)
            ensure_multi_agent_contract(env, observation)
            configure_viewer(
                env,
                camera_mode=camera_mode,
                scaling=scaling,
            )
            frame = env.render()
            render_info = inspect_render_result(
                frame,
                f"step {step_index + 1}",
                resolved_render_mode,
            )
            rollout_steps += 1
            step_records.append(
                {
                    "step": step_index + 1,
                    "joint_action": joint_action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "render_result_type": render_info["result_type"],
                    "frame_shape": render_info["frame_shape"],
                }
            )
            if terminated or truncated:
                break

        return {
            "config_path": str(config_path),
            "env_id": env_id,
            "render_mode": resolved_render_mode,
            "seed": int(seed),
            "requested_steps": int(steps),
            "executed_steps": int(rollout_steps),
            "action_mode": action_mode,
            "camera_mode": camera_mode,
            "scaling": None if scaling is None else float(scaling),
            "controlled_agent_colors": CONTROLLED_AGENT_COLORS,
            "initial_render_result_type": initial_render["result_type"],
            "initial_frame_shape": initial_render["frame_shape"],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "step_records": step_records,
            "info_keys": sorted(info.keys()) if isinstance(info, dict) else [],
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test rendering for the adversarial multi-agent racetrack environment."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the adversarial scenario config.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Maximum number of environment steps to run after reset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=21,
        help="Environment reset seed.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        help="Override render mode from config. Default keeps the config value.",
    )
    parser.add_argument(
        "--action-mode",
        type=str,
        choices=("idle", "random"),
        default="idle",
        help="How to generate the two-agent action tuple during the smoke test.",
    )
    parser.add_argument(
        "--camera-mode",
        type=str,
        choices=("auto", "first", "second", "midpoint"),
        default="midpoint",
        help="Which camera target to use for rendering.",
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=2.0,
        help="Viewer scaling. Smaller values zoom out more. Use 0 or negative to keep env default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_render_smoke_test(
        config_path=args.config_path.resolve(),
        steps=args.steps,
        seed=args.seed,
        render_mode=args.render_mode,
        action_mode=args.action_mode,
        camera_mode=args.camera_mode,
        scaling=None if args.scaling is None or args.scaling <= 0 else args.scaling,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

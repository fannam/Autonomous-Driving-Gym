from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

try:
    from AlphaZeroMetaAdversarial.core.game import classify_terminal_state
    from AlphaZeroMetaAdversarial.core.runtime_config import (
        get_scenario_config_path,
        load_runtime_config,
    )
    from AlphaZeroMetaAdversarial.core.settings import ZeroSumConfig, load_stage_config
    from AlphaZeroMetaAdversarial.environment.config import build_env_spec
except ModuleNotFoundError:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))
    from AlphaZeroMetaAdversarial.core.game import classify_terminal_state
    from AlphaZeroMetaAdversarial.core.runtime_config import (
        get_scenario_config_path,
        load_runtime_config,
    )
    from AlphaZeroMetaAdversarial.core.settings import ZeroSumConfig, load_stage_config
    from AlphaZeroMetaAdversarial.environment.config import build_env_spec


# Avoid ALSA warnings on headless/no-audio machines.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import highway_env  # noqa: F401


CONTROLLED_AGENT_COLORS = (
    (50, 200, 0),
    (220, 30, 160),
)


class ObserverProxy:
    def __init__(self, position: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=np.float32)


@dataclass(frozen=True)
class ReplayEpisodeRecord:
    manifest_path: Path | None
    shard_path: Path
    shard_name: str
    shard_index: int | None
    worker_id: int | None
    global_worker_id: int | None
    episode_position: int
    episode_summary: dict[str, Any]
    env_id: str | None
    env_config: dict[str, Any] | None
    active_scenario: str | None
    config_path: str | None
    manifest: dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a saved meta-adversarial self-play episode from manifest/shard "
            "and render the recorded joint action sequence."
        )
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--shard-path", type=Path, default=None)
    parser.add_argument("--shard-name", type=str, default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--episode-position", type=int, default=None)
    parser.add_argument("--list-episodes", action="store_true")
    parser.add_argument(
        "--search-root",
        dest="search_roots",
        action="append",
        type=Path,
        default=[],
        help="Extra root used when resolving shard paths from a manifest.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("rgb_array", "human"),
        default="human",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("midpoint", "first", "second", "auto"),
        default="midpoint",
    )
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--camera-padding-m", type=float, default=18.0)
    parser.add_argument("--min-scaling", type=float, default=1.5)
    parser.add_argument("--fixed-scaling", type=float, default=None)
    parser.add_argument("--screen-width", type=int, default=None)
    parser.add_argument("--screen-height", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-frames-dir", type=Path, default=None)
    return parser.parse_args()


def discover_manifest_paths(search_roots: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    for root in search_roots:
        candidate_root = Path(root).expanduser()
        if not candidate_root.exists():
            continue
        manifests.extend(path.resolve() for path in candidate_root.rglob("manifest.json"))
    return sorted(set(manifests))


def resolve_manifest_path(
    manifest_path: Path | None,
    *,
    search_roots: list[Path],
) -> Path | None:
    if manifest_path is None:
        manifests = discover_manifest_paths(search_roots)
        return max(manifests, key=lambda path: path.stat().st_mtime) if manifests else None

    candidate = manifest_path.expanduser().resolve()
    if candidate.is_dir():
        manifests = discover_manifest_paths([candidate])
        if not manifests:
            raise FileNotFoundError(f"No manifest.json found under directory: {candidate}")
        return max(manifests, key=lambda path: path.stat().st_mtime)
    if not candidate.exists():
        raise FileNotFoundError(f"Manifest path does not exist: {candidate}")
    return candidate


def load_manifest(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(loaded).__name__}.")
    return loaded


def flatten_manifest_shards(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if "shards" in manifest:
        return [dict(item, manifest_kind="single_worker") for item in manifest.get("shards", [])]

    rows: list[dict[str, Any]] = []
    for worker in manifest.get("workers", []):
        worker_meta = {
            "worker_id": worker.get("worker_id"),
            "global_worker_id": worker.get("global_worker_id"),
            "device": worker.get("device"),
        }
        for shard in worker.get("shards", []):
            row = dict(shard)
            for key, value in worker_meta.items():
                if key not in row and value is not None:
                    row[key] = value
            row["manifest_kind"] = "multi_worker"
            rows.append(row)
    return rows


def resolve_shard_path(
    *,
    manifest_path: Path,
    shard_entry: dict[str, Any],
    search_roots: list[Path],
) -> Path:
    raw_path = Path(str(shard_entry["path"])).expanduser()
    normalized_roots = [Path(root).expanduser().resolve() for root in search_roots]

    candidate_paths: list[Path] = []
    if raw_path.is_absolute():
        candidate_paths.append(raw_path)
    else:
        candidate_paths.append((manifest_path.parent / raw_path).resolve())
    candidate_paths.append((manifest_path.parent / raw_path.name).resolve())

    for root in normalized_roots:
        if raw_path.is_absolute():
            candidate_paths.append((root / raw_path.name).resolve())
        else:
            candidate_paths.append((root / raw_path).resolve())
            candidate_paths.append((root / raw_path.name).resolve())

    seen: set[Path] = set()
    for candidate in candidate_paths:
        normalized = candidate.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.exists():
            return normalized

    checked_paths = "\n".join(f"- {path}" for path in sorted(seen, key=str))
    raise FileNotFoundError(
        "Could not resolve shard path from manifest.\n"
        f"Manifest: {manifest_path}\n"
        f"Shard path entry: {raw_path}\n"
        f"Checked candidates:\n{checked_paths}"
    )


def load_shard_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict shard payload in {path}, got {type(payload).__name__}.")
    return payload


def normalize_joint_actions(raw_joint_actions: Any) -> list[tuple[int, int]]:
    if not isinstance(raw_joint_actions, (list, tuple)):
        return []

    normalized: list[tuple[int, int]] = []
    for item in raw_joint_actions:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            normalized.append((int(item[0]), int(item[1])))
        except (TypeError, ValueError):
            continue
    return normalized


def build_episode_records(
    *,
    manifest_path: Path | None,
    shard_path: Path | None,
    shard_name: str | None,
    search_roots: list[Path],
) -> list[ReplayEpisodeRecord]:
    records: list[ReplayEpisodeRecord] = []

    manifest: dict[str, Any] | None = None
    selected_shard_paths: list[tuple[Path, dict[str, Any] | None]] = []

    if manifest_path is not None:
        manifest = load_manifest(manifest_path)
        shard_entries = flatten_manifest_shards(manifest)
        if shard_name is not None:
            shard_entries = [
                entry
                for entry in shard_entries
                if Path(str(entry.get("path", ""))).name == shard_name
            ]
            if not shard_entries:
                raise FileNotFoundError(
                    f"Shard {shard_name!r} was not found in manifest {manifest_path}."
                )
        if shard_path is not None:
            selected_shard_paths.append((shard_path.expanduser().resolve(), None))
        else:
            for shard_entry in shard_entries:
                selected_shard_paths.append(
                    (
                        resolve_shard_path(
                            manifest_path=manifest_path,
                            shard_entry=shard_entry,
                            search_roots=search_roots,
                        ),
                        shard_entry,
                    )
                )
    elif shard_path is not None:
        selected_shard_paths.append((shard_path.expanduser().resolve(), None))
    else:
        raise ValueError("Provide at least one of --manifest-path or --shard-path.")

    for selected_path, shard_entry in selected_shard_paths:
        payload = load_shard_payload(selected_path)
        episodes = list(payload.get("episodes", []))
        for episode_position, episode_summary in enumerate(episodes):
            if not isinstance(episode_summary, dict):
                continue
            records.append(
                ReplayEpisodeRecord(
                    manifest_path=manifest_path,
                    shard_path=selected_path,
                    shard_name=selected_path.name,
                    shard_index=payload.get("shard_index", (shard_entry or {}).get("shard_index")),
                    worker_id=payload.get("worker_id", (shard_entry or {}).get("worker_id")),
                    global_worker_id=payload.get(
                        "global_worker_id",
                        (shard_entry or {}).get("global_worker_id"),
                    ),
                    episode_position=int(episode_position),
                    episode_summary=episode_summary,
                    env_id=payload.get("env_id") or (manifest or {}).get("env_id"),
                    env_config=payload.get("env_config") or (manifest or {}).get("env_config"),
                    active_scenario=payload.get("active_scenario") or (manifest or {}).get("active_scenario"),
                    config_path=payload.get("config_path") or (manifest or {}).get("config_path"),
                    manifest=manifest,
                )
            )
    return records


def select_episode_record(
    records: list[ReplayEpisodeRecord],
    *,
    episode_index: int | None,
    episode_position: int | None,
) -> ReplayEpisodeRecord:
    if not records:
        raise ValueError("No episodes were found in the provided replay source.")

    if episode_index is not None and episode_position is not None:
        raise ValueError("Use either --episode-index or --episode-position, not both.")

    if episode_index is not None:
        matches = [
            record
            for record in records
            if int(record.episode_summary.get("episode_index", -1)) == int(episode_index)
        ]
        if not matches:
            available = sorted(
                {
                    int(record.episode_summary["episode_index"])
                    for record in records
                    if "episode_index" in record.episode_summary
                }
            )
            raise ValueError(
                f"Episode index {episode_index} was not found. "
                f"Available episode_index values: {available}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Episode index {episode_index} matched multiple records. "
                "Narrow the selection with --shard-name or --shard-path."
            )
        return matches[0]

    sorted_records = sorted(
        records,
        key=lambda record: (
            int(record.episode_summary.get("episode_index", 10**12)),
            record.global_worker_id if record.global_worker_id is not None else 10**12,
            record.worker_id if record.worker_id is not None else 10**12,
            record.shard_index if record.shard_index is not None else 10**12,
            record.episode_position,
            record.shard_name,
        ),
    )
    if episode_position is None:
        return sorted_records[0]
    if episode_position < 0 or episode_position >= len(sorted_records):
        raise IndexError(
            f"--episode-position={episode_position} is outside the valid range "
            f"[0, {len(sorted_records) - 1}]."
        )
    return sorted_records[int(episode_position)]


def build_manifest_env_overrides(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(manifest, dict):
        return None
    raw_args = manifest.get("args")
    if not isinstance(raw_args, dict):
        return None

    overrides: dict[str, Any] = {}
    for key in ("duration", "vehicles_count", "policy_frequency", "simulation_frequency"):
        value = raw_args.get(key)
        if value is not None:
            overrides[key] = int(value)
    return overrides or None


def resolve_replay_config_path(
    *,
    active_scenario: str | None,
    config_path: str | None,
) -> Path | None:
    if config_path:
        candidate = Path(config_path).expanduser().resolve()
        if candidate.exists():
            return candidate
    if active_scenario:
        return get_scenario_config_path(str(active_scenario))
    return None


def build_env_metadata(
    record: ReplayEpisodeRecord,
    *,
    render_mode: str,
    screen_width: int | None,
    screen_height: int | None,
) -> tuple[str, dict[str, Any]]:
    if record.env_config is not None and record.env_id is not None:
        env_config = copy.deepcopy(record.env_config)
        env_id = str(record.env_id)
    else:
        resolved_config_path = resolve_replay_config_path(
            active_scenario=record.active_scenario,
            config_path=record.config_path,
        )
        env_spec = build_env_spec(
            stage="self_play",
            scenario_name=record.active_scenario,
            config_path=resolved_config_path,
            env_name=record.env_id,
            render_mode=render_mode,
            env_config_overrides=build_manifest_env_overrides(record.manifest),
        )
        env_id = env_spec.env_id
        env_config = env_spec.config

    if screen_width is not None:
        env_config["screen_width"] = int(screen_width)
    if screen_height is not None:
        env_config["screen_height"] = int(screen_height)
    if render_mode == "rgb_array":
        env_config["offscreen_rendering"] = True

    return env_id, env_config


def build_zero_sum_config(record: ReplayEpisodeRecord) -> ZeroSumConfig | None:
    manifest = record.manifest or {}
    raw_config = manifest.get("config")
    if isinstance(raw_config, dict):
        zero_sum = raw_config.get("zero_sum")
        if isinstance(zero_sum, dict):
            return ZeroSumConfig.from_dict(zero_sum)

    resolved_config_path = resolve_replay_config_path(
        active_scenario=record.active_scenario,
        config_path=record.config_path,
    )
    if resolved_config_path is None:
        return None
    loaded_runtime = load_runtime_config(resolved_config_path)
    replay_config = load_stage_config(
        "self_play",
        scenario_name=record.active_scenario,
        raw_config=loaded_runtime,
    )
    return replay_config.zero_sum


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
        raise RuntimeError("Expected MultiAgentAction with 2 per-agent action types.")
    return agents_action_types[0], agents_action_types[1]


def colorize_controlled_vehicles(env) -> None:
    ego_vehicle, npc_vehicle = get_controlled_vehicles(env)
    ego_vehicle.color = CONTROLLED_AGENT_COLORS[0]
    npc_vehicle.color = CONTROLLED_AGENT_COLORS[1]


def describe_observation(observation: Any) -> str:
    if isinstance(observation, tuple):
        return "tuple(" + ", ".join(
            f"agent_{index}({describe_observation(item)})"
            for index, item in enumerate(observation)
        ) + ")"
    if isinstance(observation, dict):
        return "dict(" + ", ".join(
            f"{key}:shape={np.asarray(value).shape},dtype={np.asarray(value).dtype}"
            for key, value in observation.items()
        ) + ")"
    array = np.asarray(observation)
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


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError(f"Expected HxWxC frame, got shape={array.shape!r}.")
    if array.dtype == np.uint8:
        return array
    return np.clip(array, 0, 255).astype(np.uint8)


def save_frame(frame: np.ndarray, output_dir: Path, frame_index: int) -> Path:
    if Image is None:
        raise RuntimeError("Saving frames requires Pillow. Install it or omit --save-frames-dir.")
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
    if render_mode != "human" or fps <= 0.0:
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
) -> None:
    ego_action_type, npc_action_type = get_agent_action_types(env)
    ego_label = action_label(ego_action_type, joint_action[0])
    npc_label = action_label(npc_action_type, joint_action[1])
    print(
        f"[step {step_index}] "
        f"joint_action=({joint_action[0]}:{ego_label}, {joint_action[1]}:{npc_label}) "
        f"reward={reward} terminated={terminated} truncated={truncated}"
    )


def print_episode_listing(records: list[ReplayEpisodeRecord]) -> None:
    print(
        "episode_index episode_position shard_name worker global_worker steps "
        "joint_actions outcome"
    )
    for record in sorted(
        records,
        key=lambda item: (
            int(item.episode_summary.get("episode_index", 10**12)),
            item.global_worker_id if item.global_worker_id is not None else 10**12,
            item.worker_id if item.worker_id is not None else 10**12,
            item.shard_index if item.shard_index is not None else 10**12,
            item.episode_position,
            item.shard_name,
        ),
    ):
        joint_actions = normalize_joint_actions(record.episode_summary.get("joint_actions"))
        print(
            f"{int(record.episode_summary.get('episode_index', -1)):>12} "
            f"{record.episode_position:>16} "
            f"{record.shard_name} "
            f"{str(record.worker_id):>6} "
            f"{str(record.global_worker_id):>13} "
            f"{int(record.episode_summary.get('steps', 0)):>5} "
            f"{len(joint_actions):>13} "
            f"{record.episode_summary.get('outcome_reason')}"
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

    search_roots = [Path(root).expanduser().resolve() for root in args.search_roots]
    manifest_path = (
        resolve_manifest_path(
            args.manifest_path,
            search_roots=search_roots or [Path.cwd()],
        )
        if (args.manifest_path is not None or args.shard_path is None)
        else None
    )
    records = build_episode_records(
        manifest_path=manifest_path,
        shard_path=args.shard_path,
        shard_name=args.shard_name,
        search_roots=search_roots,
    )

    if args.list_episodes:
        print_episode_listing(records)
        return 0

    record = select_episode_record(
        records,
        episode_index=args.episode_index,
        episode_position=args.episode_position,
    )
    joint_actions = normalize_joint_actions(record.episode_summary.get("joint_actions"))
    if not joint_actions:
        raise ValueError(
            "Selected episode does not contain `joint_actions`. "
            "Replay requires shards generated after the action-history upgrade."
        )

    env_seed = int(record.episode_summary.get("env_seed", record.episode_summary.get("seed", 21)))
    env_id, env_config = build_env_metadata(
        record,
        render_mode=args.render_mode,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
    )
    zero_sum_config = build_zero_sum_config(record)
    max_steps = len(joint_actions) if args.max_steps is None else min(len(joint_actions), int(args.max_steps))

    print(
        "[meta-adversarial-replay] "
        f"manifest={manifest_path} "
        f"shard={record.shard_path} "
        f"episode_index={record.episode_summary.get('episode_index')} "
        f"episode_position={record.episode_position} "
        f"env_seed={env_seed} "
        f"saved_steps={record.episode_summary.get('steps')} "
        f"replay_steps={max_steps} "
        f"render_mode={args.render_mode}"
    )
    print(
        f"[episode] outcome={record.episode_summary.get('outcome_reason')} "
        f"ego_value={record.episode_summary.get('ego_value')} "
        f"npc_value={record.episode_summary.get('npc_value')} "
        f"joint_actions={len(joint_actions)}"
    )
    print(f"[env] env_id={env_id}")
    print(f"[env] config={env_config}")

    env = gym.make(
        env_id,
        render_mode=args.render_mode,
        config=copy.deepcopy(env_config),
    )
    frame_index = 0
    terminated = False
    truncated = False
    last_reward: Any = None
    replay_outcome = None

    try:
        observation, info = env.reset(seed=env_seed)
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

        for step_index, joint_action in enumerate(joint_actions[:max_steps], start=1):
            observation, last_reward, terminated, truncated, info = env.step(joint_action)
            print_step_info(
                env,
                step_index=step_index,
                joint_action=joint_action,
                reward=last_reward,
                terminated=terminated,
                truncated=truncated,
            )
            print(f"[step {step_index}] obs={describe_observation(observation)}")
            if info:
                print(f"[step {step_index}] info_keys={sorted(info.keys())}")
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

            if terminated or truncated:
                print(f"[stop] environment ended at replay step={step_index}")
                break
        if zero_sum_config is not None:
            replay_outcome = classify_terminal_state(env, zero_sum_config)
    finally:
        env.close()

    if replay_outcome is not None:
        print(
            f"[replay-outcome] terminal={replay_outcome.terminal} "
            f"reason={replay_outcome.reason} "
            f"ego_value={replay_outcome.ego_value:.2f} "
            f"npc_value={replay_outcome.npc_value:.2f}"
        )
    else:
        print("[replay-outcome] zero_sum_config unavailable; skipped outcome classification")

    if max_steps < len(joint_actions):
        print(
            f"[summary] replay truncated to {max_steps} / {len(joint_actions)} saved joint actions"
        )
    else:
        print(f"[summary] replayed all {len(joint_actions)} saved joint actions")
    print(
        f"[summary] frames={frame_index} terminated={terminated} truncated={truncated} "
        f"last_reward={last_reward}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

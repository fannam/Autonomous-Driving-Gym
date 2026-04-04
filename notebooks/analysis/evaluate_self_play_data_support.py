from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


META_ACTION_LABELS = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}

BASE_TARGET_VECTOR_NAMES = [
    "dx",
    "dy",
    "dvx",
    "dvy",
    "sin_bearing",
    "cos_bearing",
]


def discover_manifest_paths(search_roots: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        manifests.extend(path.resolve() for path in root.rglob("manifest.json"))
    return sorted(set(manifests))


def resolve_manifest_path(manifest_path: str | None, search_roots: list[Path]) -> Path:
    if manifest_path:
        candidate = Path(manifest_path).expanduser().resolve()
        if candidate.is_dir():
            manifests = discover_manifest_paths([candidate])
            if not manifests:
                raise FileNotFoundError(f"No manifest.json found under directory: {candidate}")
            return max(manifests, key=lambda path: path.stat().st_mtime)
        if not candidate.exists():
            raise FileNotFoundError(f"Manifest path does not exist: {candidate}")
        return candidate

    manifests = discover_manifest_paths(search_roots)
    if not manifests:
        roots = "\n".join(f"- {root}" for root in search_roots)
        raise FileNotFoundError(
            "Could not discover any manifest.json automatically. Set MANIFEST_PATH explicitly.\n"
            f"Search roots:\n{roots}"
        )
    return max(manifests, key=lambda path: path.stat().st_mtime)


def load_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def flatten_manifest_shards(manifest: dict) -> list[dict]:
    if "shards" in manifest:
        return [dict(item, manifest_kind="single_worker") for item in manifest.get("shards", [])]

    rows: list[dict] = []
    for worker in manifest.get("workers", []):
        worker_meta = {
            "worker_id": worker.get("worker_id"),
            "global_worker_id": worker.get("global_worker_id"),
            "device": worker.get("device"),
            "episodes_per_worker": worker.get("episodes_per_worker"),
            "worker_total_samples": worker.get("total_samples"),
        }
        for shard in worker.get("shards", []):
            row = dict(shard)
            for key, value in worker_meta.items():
                if key not in row and value is not None:
                    row[key] = value
            row["manifest_kind"] = "multi_worker"
            rows.append(row)
    return rows


def resolve_shard_path(manifest_path: Path, shard_entry: dict) -> Path:
    raw_path = Path(shard_entry["path"]).expanduser()
    if raw_path.is_absolute():
        return raw_path
    return (manifest_path.parent / raw_path).resolve()


def load_raw_runtime_config(manifest: dict) -> dict[str, Any] | None:
    config_path = manifest.get("config_path")
    if not config_path:
        return None
    path = Path(config_path).expanduser()
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    return loaded if isinstance(loaded, dict) else None


def infer_target_vector_names(manifest: dict, target_vector_dim: int | None = None) -> list[str]:
    tensor_cfg = (manifest.get("config") or {}).get("tensor") or {}
    names = list(BASE_TARGET_VECTOR_NAMES)
    if tensor_cfg.get("include_role_bit"):
        names.append("role_bit")
    if tensor_cfg.get("include_target_type_bit"):
        names.append("target_type_bit")
    if target_vector_dim is None or len(names) == int(target_vector_dim):
        return names
    return [f"target_{index}" for index in range(int(target_vector_dim))]


def infer_roles(target_vectors: np.ndarray, target_names: list[str]) -> np.ndarray:
    if "role_bit" not in target_names or target_vectors.size == 0:
        return np.full(target_vectors.shape[0], "unknown", dtype=object)
    role_index = target_names.index("role_bit")
    role_values = np.asarray(target_vectors[:, role_index], dtype=np.float32)
    return np.where(role_values > 0.0, "ego", np.where(role_values < 0.0, "npc", "unknown"))


def policy_entropy_batch(policy_array: np.ndarray) -> np.ndarray:
    if policy_array.size == 0:
        return np.empty((0,), dtype=np.float64)
    probs = np.clip(np.asarray(policy_array, dtype=np.float64), 1e-12, 1.0)
    return -(probs * np.log(probs)).sum(axis=1)


def infer_policy_format(payload: dict) -> str:
    declared = payload.get("policy_format")
    if isinstance(declared, str) and declared:
        return declared
    if "policies" in payload:
        return "flat_meta"
    if "accelerate_policies" in payload and "steering_policies" in payload:
        return "factorized_axes"
    raise ValueError("Could not infer policy format from shard payload.")


def _format_numeric_label(value: float) -> str:
    return f"{float(value):+.2f}"


def infer_factorized_axis_info(
    manifest: dict,
    *,
    accelerate_dim: int,
    steering_dim: int,
) -> dict[str, Any]:
    config = manifest.get("config") or {}
    raw_config = load_raw_runtime_config(manifest) or {}
    action_cfg = (
        raw_config.get("environment", {})
        .get("config", {})
        .get("action", {})
        .get("action_config", {})
    )

    axis_0_dim = int(config.get("n_action_axis_0", accelerate_dim))
    axis_1_dim = int(config.get("n_action_axis_1", steering_dim))
    if axis_0_dim != int(accelerate_dim) or axis_1_dim != int(steering_dim):
        axis_0_dim = int(accelerate_dim)
        axis_1_dim = int(steering_dim)

    acceleration_range = action_cfg.get("acceleration_range")
    steering_range = action_cfg.get("steering_range")

    if isinstance(acceleration_range, (list, tuple)) and len(acceleration_range) == 2:
        axis_0_values = np.linspace(
            float(acceleration_range[0]),
            float(acceleration_range[1]),
            axis_0_dim,
            dtype=np.float64,
        )
    else:
        axis_0_values = np.linspace(-1.0, 1.0, axis_0_dim, dtype=np.float64)

    if isinstance(steering_range, (list, tuple)) and len(steering_range) == 2:
        axis_1_values = np.linspace(
            float(steering_range[0]),
            float(steering_range[1]),
            axis_1_dim,
            dtype=np.float64,
        )
    else:
        axis_1_values = np.linspace(-1.0, 1.0, axis_1_dim, dtype=np.float64)

    axis_0_labels = [
        f"a{index}:{_format_numeric_label(value)}"
        for index, value in enumerate(axis_0_values.tolist())
    ]
    axis_1_labels = [
        f"s{index}:{_format_numeric_label(value)}"
        for index, value in enumerate(axis_1_values.tolist())
    ]

    return {
        "n_action_axis_0": axis_0_dim,
        "n_action_axis_1": axis_1_dim,
        "axis_0_values": axis_0_values,
        "axis_1_values": axis_1_values,
        "axis_0_labels": axis_0_labels,
        "axis_1_labels": axis_1_labels,
        "axis_0_center": axis_0_dim // 2,
        "axis_1_center": axis_1_dim // 2,
    }


def summarize_manifest(analysis: dict) -> pd.DataFrame:
    manifest = analysis["manifest"]
    shard_df = analysis["shard_df"]
    sample_df = analysis["sample_df"]
    episode_df = analysis["episode_df"]
    return pd.DataFrame(
        [
            {
                "manifest_path": str(analysis["manifest_path"]),
                "active_scenario": manifest.get("active_scenario"),
                "env_id": manifest.get("env_id"),
                "manifest_kind": "multi_worker" if "workers" in manifest else "single_worker",
                "policy_format": analysis["policy_format"],
                "total_shards_manifest": manifest.get("total_shards"),
                "total_shards_loaded": int(shard_df.shape[0]),
                "total_samples_manifest": manifest.get("total_samples"),
                "total_samples_loaded": int(sample_df.shape[0]),
                "total_episodes_loaded": int(episode_df.shape[0]),
            }
        ]
    )


def _build_sample_frame_flat_meta(
    *,
    policies_np: np.ndarray,
    values_np: np.ndarray,
    roles: np.ndarray,
    worker_id: int | None,
    global_worker_id: int | None,
    shard_index: int | None,
) -> pd.DataFrame:
    action_ids = policies_np.argmax(axis=1) if policies_np.size else np.empty((0,), dtype=np.int64)
    action_labels = [META_ACTION_LABELS.get(int(action_id), f"action_{int(action_id)}") for action_id in action_ids]
    policy_entropy = policy_entropy_batch(policies_np)
    max_policy_prob = policies_np.max(axis=1) if policies_np.size else np.empty((0,), dtype=np.float32)
    policy_row_sum = policies_np.sum(axis=1) if policies_np.size else np.empty((0,), dtype=np.float32)
    return pd.DataFrame(
        {
            "policy_format": "flat_meta",
            "worker_id": worker_id,
            "global_worker_id": global_worker_id,
            "shard_index": shard_index,
            "role": roles,
            "action_id": action_ids,
            "action_label": action_labels,
            "policy_entropy": policy_entropy,
            "max_policy_prob": max_policy_prob,
            "policy_row_sum": policy_row_sum,
            "value": values_np,
        }
    )


def _build_sample_frame_factorized_axes(
    *,
    accelerate_np: np.ndarray,
    steering_np: np.ndarray,
    values_np: np.ndarray,
    roles: np.ndarray,
    worker_id: int | None,
    global_worker_id: int | None,
    shard_index: int | None,
    axis_info: dict[str, Any],
) -> pd.DataFrame:
    accelerate_action_id = (
        accelerate_np.argmax(axis=1) if accelerate_np.size else np.empty((0,), dtype=np.int64)
    )
    steering_action_id = (
        steering_np.argmax(axis=1) if steering_np.size else np.empty((0,), dtype=np.int64)
    )
    accelerate_labels = [axis_info["axis_0_labels"][int(index)] for index in accelerate_action_id]
    steering_labels = [axis_info["axis_1_labels"][int(index)] for index in steering_action_id]
    action_id = accelerate_action_id * int(axis_info["n_action_axis_1"]) + steering_action_id
    action_label = [
        f"{accelerate_label} | {steering_label}"
        for accelerate_label, steering_label in zip(accelerate_labels, steering_labels)
    ]

    accelerate_row_sum = accelerate_np.sum(axis=1) if accelerate_np.size else np.empty((0,), dtype=np.float32)
    steering_row_sum = steering_np.sum(axis=1) if steering_np.size else np.empty((0,), dtype=np.float32)
    accelerate_entropy = policy_entropy_batch(accelerate_np)
    steering_entropy = policy_entropy_batch(steering_np)
    max_policy_prob = (
        accelerate_np.max(axis=1) * steering_np.max(axis=1)
        if accelerate_np.size and steering_np.size
        else np.empty((0,), dtype=np.float32)
    )

    return pd.DataFrame(
        {
            "policy_format": "factorized_axes",
            "worker_id": worker_id,
            "global_worker_id": global_worker_id,
            "shard_index": shard_index,
            "role": roles,
            "action_id": action_id,
            "action_label": action_label,
            "accelerate_action_id": accelerate_action_id,
            "steering_action_id": steering_action_id,
            "accelerate_action_label": accelerate_labels,
            "steering_action_label": steering_labels,
            "accelerate_entropy": accelerate_entropy,
            "steering_entropy": steering_entropy,
            "policy_entropy": accelerate_entropy + steering_entropy,
            "accelerate_row_sum": accelerate_row_sum,
            "steering_row_sum": steering_row_sum,
            "policy_row_sum": accelerate_row_sum * steering_row_sum,
            "max_policy_prob": max_policy_prob,
            "is_center_joint": (
                (accelerate_action_id == int(axis_info["axis_0_center"]))
                & (steering_action_id == int(axis_info["axis_1_center"]))
            ),
            "value": values_np,
        }
    )


def analyze_manifest(manifest_path: Path) -> dict:
    manifest = load_manifest(manifest_path)
    shard_entries = flatten_manifest_shards(manifest)
    if not shard_entries:
        raise ValueError(f"Manifest contains no shards: {manifest_path}")

    shard_rows: list[dict] = []
    episode_rows: list[dict] = []
    sample_frames: list[pd.DataFrame] = []
    target_names: list[str] | None = None
    target_accumulators: dict[str, dict[str, Any]] = {}
    policy_formats: set[str] = set()
    factorized_axis_info: dict[str, Any] | None = None

    for shard_entry in shard_entries:
        shard_path = resolve_shard_path(manifest_path, shard_entry)
        payload = torch.load(shard_path, map_location="cpu")

        states = payload["states"]
        target_vectors = payload["target_vectors"]
        values = payload["values"]
        episodes = list(payload.get("episodes", []))
        policy_format = infer_policy_format(payload)
        policy_formats.add(policy_format)

        target_np = target_vectors.detach().cpu().numpy()
        values_np = values.detach().cpu().numpy().reshape(-1)

        if target_names is None:
            target_names = infer_target_vector_names(manifest, target_np.shape[1])
        if target_np.shape[1] != len(target_names):
            raise ValueError(
                f"Target-vector dimension mismatch in {shard_path}: "
                f"expected {len(target_names)}, got {target_np.shape[1]}"
            )

        roles = infer_roles(target_np, target_names)
        worker_id = payload.get("worker_id", shard_entry.get("worker_id"))
        global_worker_id = payload.get("global_worker_id", shard_entry.get("global_worker_id"))
        shard_index = payload.get("shard_index", shard_entry.get("shard_index"))

        row: dict[str, Any] = {
            "path": str(shard_path),
            "manifest_kind": shard_entry.get("manifest_kind"),
            "worker_id": worker_id,
            "global_worker_id": global_worker_id,
            "shard_index": shard_index,
            "policy_format": policy_format,
            "episode_count": int(payload.get("episode_count", len(episodes))),
            "sample_count": int(payload.get("sample_count", len(values_np))),
            "episode_sample_count_sum": int(sum(int(item.get("sample_count", 0)) for item in episodes)),
            "state_shape": tuple(int(item) for item in states.shape),
            "target_shape": tuple(int(item) for item in target_vectors.shape),
            "value_shape": tuple(int(item) for item in values.shape),
            "state_nonfinite": int((~torch.isfinite(states)).sum().item()),
            "target_nonfinite": int((~torch.isfinite(target_vectors)).sum().item()),
            "value_nonfinite": int((~torch.isfinite(values)).sum().item()),
            "ego_samples": int((roles == "ego").sum()),
            "npc_samples": int((roles == "npc").sum()),
            "unknown_role_samples": int((roles == "unknown").sum()),
            "value_min": float(values_np.min()) if len(values_np) else np.nan,
            "value_max": float(values_np.max()) if len(values_np) else np.nan,
        }

        if policy_format == "flat_meta":
            policies = payload["policies"]
            policies_np = policies.detach().cpu().numpy()
            row.update(
                {
                    "policy_shape": tuple(int(item) for item in policies.shape),
                    "policy_nonfinite": int((~torch.isfinite(policies)).sum().item()),
                    "policy_row_sum_min": float(policies_np.sum(axis=1).min()) if len(policies_np) else np.nan,
                    "policy_row_sum_max": float(policies_np.sum(axis=1).max()) if len(policies_np) else np.nan,
                }
            )
            sample_frames.append(
                _build_sample_frame_flat_meta(
                    policies_np=policies_np,
                    values_np=values_np,
                    roles=roles,
                    worker_id=worker_id,
                    global_worker_id=global_worker_id,
                    shard_index=shard_index,
                )
            )
        elif policy_format == "factorized_axes":
            accelerate_policies = payload["accelerate_policies"]
            steering_policies = payload["steering_policies"]
            accelerate_np = accelerate_policies.detach().cpu().numpy()
            steering_np = steering_policies.detach().cpu().numpy()

            if factorized_axis_info is None:
                factorized_axis_info = infer_factorized_axis_info(
                    manifest,
                    accelerate_dim=accelerate_np.shape[1],
                    steering_dim=steering_np.shape[1],
                )

            accelerate_row_sum = accelerate_np.sum(axis=1) if len(accelerate_np) else np.empty((0,))
            steering_row_sum = steering_np.sum(axis=1) if len(steering_np) else np.empty((0,))
            row.update(
                {
                    "accelerate_policy_shape": tuple(int(item) for item in accelerate_policies.shape),
                    "steering_policy_shape": tuple(int(item) for item in steering_policies.shape),
                    "accelerate_policy_nonfinite": int((~torch.isfinite(accelerate_policies)).sum().item()),
                    "steering_policy_nonfinite": int((~torch.isfinite(steering_policies)).sum().item()),
                    "accelerate_row_sum_min": float(accelerate_row_sum.min()) if len(accelerate_row_sum) else np.nan,
                    "accelerate_row_sum_max": float(accelerate_row_sum.max()) if len(accelerate_row_sum) else np.nan,
                    "steering_row_sum_min": float(steering_row_sum.min()) if len(steering_row_sum) else np.nan,
                    "steering_row_sum_max": float(steering_row_sum.max()) if len(steering_row_sum) else np.nan,
                    "policy_row_sum_min": (
                        float((accelerate_row_sum * steering_row_sum).min())
                        if len(accelerate_row_sum) and len(steering_row_sum)
                        else np.nan
                    ),
                    "policy_row_sum_max": (
                        float((accelerate_row_sum * steering_row_sum).max())
                        if len(accelerate_row_sum) and len(steering_row_sum)
                        else np.nan
                    ),
                }
            )
            sample_frames.append(
                _build_sample_frame_factorized_axes(
                    accelerate_np=accelerate_np,
                    steering_np=steering_np,
                    values_np=values_np,
                    roles=roles,
                    worker_id=worker_id,
                    global_worker_id=global_worker_id,
                    shard_index=shard_index,
                    axis_info=factorized_axis_info,
                )
            )
        else:
            raise ValueError(f"Unsupported policy format: {policy_format}")

        shard_rows.append(row)

        for role_name in ("ego", "npc", "unknown"):
            mask = roles == role_name
            if not np.any(mask):
                continue
            if role_name not in target_accumulators:
                target_accumulators[role_name] = {
                    "count": 0,
                    "sum": np.zeros(target_np.shape[1], dtype=np.float64),
                    "sumsq": np.zeros(target_np.shape[1], dtype=np.float64),
                }
            accumulator = target_accumulators[role_name]
            selected = target_np[mask].astype(np.float64)
            accumulator["count"] = int(accumulator["count"]) + int(selected.shape[0])
            accumulator["sum"] = np.asarray(accumulator["sum"]) + selected.sum(axis=0)
            accumulator["sumsq"] = np.asarray(accumulator["sumsq"]) + np.square(selected).sum(axis=0)

        offset = 0
        for episode in episodes:
            episode_sample_count = int(episode.get("sample_count", 0))
            episode_roles = roles[offset : offset + episode_sample_count]
            policy_modes = episode.get("policy_modes")
            episode_row = dict(episode)
            episode_row.update(
                {
                    "path": str(shard_path),
                    "worker_id": worker_id,
                    "global_worker_id": global_worker_id,
                    "shard_index": shard_index,
                    "policy_format": policy_format,
                    "policy_modes_label": (
                        repr(tuple(policy_modes))
                        if isinstance(policy_modes, (list, tuple))
                        else repr(policy_modes)
                    ),
                    "sample_start": int(offset),
                    "sample_end": int(offset + episode_sample_count),
                    "ego_samples": int((episode_roles == "ego").sum()),
                    "npc_samples": int((episode_roles == "npc").sum()),
                    "unknown_role_samples": int((episode_roles == "unknown").sum()),
                }
            )
            episode_rows.append(episode_row)
            offset += episode_sample_count

    shard_df = pd.DataFrame(shard_rows).sort_values(
        by=["global_worker_id", "worker_id", "shard_index", "path"],
        na_position="last",
    )
    episode_df = pd.DataFrame(episode_rows).sort_values(
        by=["episode_index", "global_worker_id", "worker_id", "shard_index"],
        na_position="last",
    )
    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()

    target_summary_rows: list[dict] = []
    for role_name, accumulator in sorted(target_accumulators.items()):
        count = int(accumulator["count"])
        if count <= 0:
            continue
        sum_vector = np.asarray(accumulator["sum"], dtype=np.float64)
        sumsq_vector = np.asarray(accumulator["sumsq"], dtype=np.float64)
        mean_vector = sum_vector / count
        variance_vector = np.maximum(sumsq_vector / count - np.square(mean_vector), 0.0)
        row = {"role": role_name, "sample_count": count}
        for index, name in enumerate(target_names):
            row[f"{name}_mean"] = float(mean_vector[index])
            row[f"{name}_std"] = float(np.sqrt(variance_vector[index]))
        target_summary_rows.append(row)
    target_stats_df = pd.DataFrame(target_summary_rows)

    sorted_policy_formats = sorted(policy_formats)
    policy_format = (
        sorted_policy_formats[0]
        if len(sorted_policy_formats) == 1
        else ",".join(sorted_policy_formats)
    )

    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "target_names": target_names,
        "policy_formats": sorted_policy_formats,
        "policy_format": policy_format,
        "factorized_axis_info": factorized_axis_info,
        "shard_df": shard_df,
        "episode_df": episode_df,
        "sample_df": sample_df,
        "target_stats_df": target_stats_df,
    }

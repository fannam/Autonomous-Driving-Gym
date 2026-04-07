from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from AlphaZeroMetaAdversarial.scripts.replay import (
    ReplayEpisodeRecord,
    build_env_metadata,
    build_episode_records,
    normalize_joint_actions,
    select_episode_record,
)
from AlphaZeroMetaAdversarial.scripts.self_play_save import _flush_shard


def test_replay_selects_episode_from_manifest_and_shard(tmp_path: Path) -> None:
    shard_path = tmp_path / "worker_00_shard_000.pt"
    torch.save(
        {
            "worker_id": 0,
            "shard_index": 0,
            "env_id": "highway-v0",
            "env_config": {"duration": 5},
            "episodes": [
                {
                    "episode_index": 3,
                    "steps": 2,
                    "outcome_reason": "npc_hit_ego",
                    "joint_actions": [(1, 2), (3, 4)],
                },
                {
                    "episode_index": 4,
                    "steps": 1,
                    "outcome_reason": "ego_finished",
                    "joint_actions": [(0, 1)],
                },
            ],
        },
        shard_path,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "active_scenario": "highway_meta_adversarial",
                "env_id": "highway-v0",
                "workers": [
                    {
                        "worker_id": 0,
                        "global_worker_id": 10,
                        "shards": [
                            {
                                "path": shard_path.name,
                                "shard_index": 0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    records = build_episode_records(
        manifest_path=manifest_path,
        shard_path=None,
        shard_name=None,
        search_roots=[],
    )
    selected = select_episode_record(records, episode_index=4, episode_position=None)

    assert len(records) == 2
    assert selected.shard_path == shard_path
    assert selected.episode_position == 1
    assert normalize_joint_actions(selected.episode_summary.get("joint_actions")) == [(0, 1)]


def test_replay_build_env_metadata_falls_back_to_manifest_runtime_config() -> None:
    record = ReplayEpisodeRecord(
        manifest_path=None,
        shard_path=Path("/tmp/dummy.pt"),
        shard_name="dummy.pt",
        shard_index=0,
        worker_id=0,
        global_worker_id=0,
        episode_position=0,
        episode_summary={"episode_index": 0},
        env_id="highway-v0",
        env_config=None,
        active_scenario="highway_meta_adversarial",
        config_path=None,
        manifest={
            "active_scenario": "highway_meta_adversarial",
            "env_id": "highway-v0",
            "args": {
                "duration": 7,
                "vehicles_count": 6,
                "policy_frequency": 1,
                "simulation_frequency": 5,
            },
        },
    )

    env_id, env_config = build_env_metadata(
        record,
        render_mode="rgb_array",
        screen_width=840,
        screen_height=260,
    )

    assert env_id == "highway-v0"
    assert env_config["duration"] == 7
    assert env_config["vehicles_count"] == 6
    assert env_config["policy_frequency"] == 1
    assert env_config["simulation_frequency"] == 5
    assert env_config["screen_width"] == 840
    assert env_config["screen_height"] == 260
    assert env_config["offscreen_rendering"] is True


def test_self_play_flush_shard_persists_replay_env_metadata(tmp_path: Path) -> None:
    summary = {
        "episode_index": 0,
        "steps": 1,
        "joint_actions": [(1, 1)],
        "outcome_reason": "ongoing",
    }
    manifest_entry = _flush_shard(
        shard_examples=[
            (
                np.zeros((3, 4, 4), dtype=np.float32),
                np.zeros((8,), dtype=np.float32),
                np.asarray([0.2, 0.3, 0.1, 0.25, 0.15], dtype=np.float32),
                0.5,
            )
        ],
        shard_episode_summaries=[summary],
        shard_index=0,
        output_dir=tmp_path,
        active_scenario="highway_meta_adversarial",
        config_path=Path("/tmp/highway_meta_adversarial.yaml"),
        env_id="highway-v0",
        env_config={"duration": 9, "vehicles_count": 5},
        model_path=tmp_path / "model.pth",
        model_sha256="abc123",
        network_seed=42,
    )

    assert manifest_entry is not None
    payload = torch.load(tmp_path / "worker_00_shard_000.pt", map_location="cpu")
    assert payload["active_scenario"] == "highway_meta_adversarial"
    assert payload["config_path"] == "/tmp/highway_meta_adversarial.yaml"
    assert payload["env_id"] == "highway-v0"
    assert payload["env_config"] == {"duration": 9, "vehicles_count": 5}

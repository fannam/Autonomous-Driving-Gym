from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym

from ..core.runtime_config import (
    get_active_scenario_name,
    get_environment_config,
    get_stage_preset_config,
    merge_config_dicts,
)


def bootstrap_local_highway_env(source_root: str | Path) -> Path:
    local_highway_env_root = Path(source_root).expanduser().resolve() / "highway-env"
    if (
        local_highway_env_root.exists()
        and str(local_highway_env_root) not in sys.path
    ):
        sys.path.insert(0, str(local_highway_env_root))
    import highway_env  # noqa: F401
    return local_highway_env_root


_REPO_ROOT = Path(__file__).resolve().parents[6]
_SOURCE_ROOT = _REPO_ROOT / "source"
bootstrap_local_highway_env(_SOURCE_ROOT)


_NATIVE_REWARD_OVERRIDES: dict[str, Any] = {
    "right_lane_reward": 0.0,
}


def _disable_native_reward_shaping(env_config: dict[str, Any]) -> dict[str, Any]:
    for key, value in _NATIVE_REWARD_OVERRIDES.items():
        env_config[key] = copy.deepcopy(value)
    return env_config


@dataclass(frozen=True)
class EnvironmentSpec:
    scenario_name: str
    env_id: str
    render_mode: str | None
    config: dict[str, Any]


def build_env_spec(
    *,
    stage: str = "train",
    scenario_name: str | None = None,
    config_path: str | Path | None = None,
    env_name: str | None = None,
    render_mode: str | None = None,
    env_config_overrides: dict[str, Any] | None = None,
) -> EnvironmentSpec:
    scenario_to_use = scenario_name or get_active_scenario_name(config_path=config_path)
    environment = get_environment_config(
        stage=stage,
        scenario_name=scenario_to_use,
        config_path=config_path,
    )
    env_id = str(env_name or environment.get("env_id"))
    if not env_id:
        raise ValueError(
            f"Scenario {scenario_to_use!r} does not define a valid environment id."
        )

    base_config = environment.get("config", {})
    if not isinstance(base_config, dict):
        raise ValueError(
            f"Scenario {scenario_to_use!r} has invalid environment config payload."
        )
    env_config = copy.deepcopy(base_config)
    preset = get_stage_preset_config(
        stage=stage,
        scenario_name=scenario_to_use,
        config_path=config_path,
    )
    stage_env_overrides = preset.get("env_overrides", {})
    if stage_env_overrides:
        if not isinstance(stage_env_overrides, dict):
            raise ValueError(
                f"Stage preset env_overrides for scenario {scenario_to_use!r} must be a mapping."
            )
        env_config = merge_config_dicts(env_config, stage_env_overrides)
    if env_config_overrides:
        env_config = merge_config_dicts(env_config, env_config_overrides)
    env_config = _disable_native_reward_shaping(env_config)

    if render_mode is not None:
        render_mode_to_use = render_mode
    elif "render_mode" in preset:
        render_mode_to_use = preset.get("render_mode")
    else:
        render_mode_to_use = environment.get("render_mode")
    return EnvironmentSpec(
        scenario_name=scenario_to_use,
        env_id=env_id,
        render_mode=render_mode_to_use,
        config=env_config,
    )


def init_env(
    *,
    env_name: str | None = None,
    seed: int = 21,
    render_mode: str | None = None,
    stage: str = "train",
    scenario_name: str | None = None,
    config_path: str | Path | None = None,
    env_config_overrides: dict[str, Any] | None = None,
):
    spec = build_env_spec(
        stage=stage,
        scenario_name=scenario_name,
        config_path=config_path,
        env_name=env_name,
        render_mode=render_mode,
        env_config_overrides=env_config_overrides,
    )
    env = gym.make(
        spec.env_id,
        config=copy.deepcopy(spec.config),
        render_mode=spec.render_mode,
    )
    env.reset(seed=seed)
    return env


__all__ = ["EnvironmentSpec", "bootstrap_local_highway_env", "build_env_spec", "init_env"]

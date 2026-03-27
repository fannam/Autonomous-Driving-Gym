from __future__ import annotations

import copy
from dataclasses import dataclass

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

try:
    from core.runtime_config import (
        get_active_scenario_name,
        get_environment_config,
        merge_config_dicts,
    )
except ModuleNotFoundError as exc:
    if exc.name != "core":
        raise
    from ..core.runtime_config import (
        get_active_scenario_name,
        get_environment_config,
        merge_config_dicts,
    )


def compute_grid_shape(grid_size, grid_step):
    grid_size_arr = np.asarray(grid_size, dtype=np.float32)
    grid_step_arr = np.asarray(grid_step, dtype=np.float32)
    grid_shape = np.floor((grid_size_arr[:, 1] - grid_size_arr[:, 0]) / grid_step_arr)
    return tuple(int(axis_cells) for axis_cells in grid_shape)


def _default_environment_payload() -> dict:
    return get_environment_config(stage="self_play")


_DEFAULT_ENVIRONMENT_PAYLOAD = _default_environment_payload()
DEFAULT_OBSERVATION_FEATURES = (
    _DEFAULT_ENVIRONMENT_PAYLOAD.get("config", {})
    .get("observation", {})
    .get("features", ["presence", "on_lane", "on_road"])
)
DEFAULT_GRID_SIZE = (
    _DEFAULT_ENVIRONMENT_PAYLOAD.get("config", {})
    .get("observation", {})
    .get("grid_size", [[-50, 50], [-12, 12]])
)
DEFAULT_GRID_STEP = (
    _DEFAULT_ENVIRONMENT_PAYLOAD.get("config", {})
    .get("observation", {})
    .get("grid_step", [1.0, 1.0])
)


@dataclass(frozen=True)
class EnvironmentSpec:
    scenario_name: str
    env_id: str
    render_mode: str | None
    config: dict


def build_env_spec(
    *,
    stage: str = "self_play",
    scenario_name: str | None = None,
    env_name: str | None = None,
    render_mode: str | None = None,
    env_config_overrides: dict | None = None,
) -> EnvironmentSpec:
    resolved_scenario = scenario_name or get_active_scenario_name()
    environment = get_environment_config(
        stage=stage,
        scenario_name=resolved_scenario,
    )
    env_id = str(env_name or environment.get("env_id"))
    if not env_id:
        raise ValueError(
            f"Scenario {resolved_scenario!r} does not define a valid environment id."
        )

    base_config = environment.get("config", {})
    if not isinstance(base_config, dict):
        raise ValueError(
            f"Scenario {resolved_scenario!r} has invalid environment config payload."
        )
    env_config = copy.deepcopy(base_config)
    if env_config_overrides:
        env_config = merge_config_dicts(env_config, env_config_overrides)

    resolved_render_mode = (
        render_mode if render_mode is not None else environment.get("render_mode")
    )
    return EnvironmentSpec(
        scenario_name=resolved_scenario,
        env_id=env_id,
        render_mode=resolved_render_mode,
        config=env_config,
    )


class EnvironmentFactory:
    @staticmethod
    def default_spec(stage: str = "self_play", scenario_name: str | None = None) -> EnvironmentSpec:
        return build_env_spec(stage=stage, scenario_name=scenario_name)

    @staticmethod
    def default_config(
        vehicle_density: float | None = None,
        stage: str = "self_play",
        scenario_name: str | None = None,
    ) -> dict:
        config = copy.deepcopy(
            EnvironmentFactory.default_spec(
                stage=stage,
                scenario_name=scenario_name,
            ).config
        )
        if vehicle_density is not None and "vehicles_density" in config:
            config["vehicles_density"] = vehicle_density
        return config

    @classmethod
    def create(
        cls,
        env_name: str | None = None,
        vehicle_density: float | None = None,
        seed: int = 21,
        render_mode: str | None = None,
        stage: str = "self_play",
        scenario_name: str | None = None,
        env_config_overrides: dict | None = None,
    ):
        spec = build_env_spec(
            stage=stage,
            scenario_name=scenario_name,
            env_name=env_name,
            render_mode=render_mode,
            env_config_overrides=env_config_overrides,
        )
        env_config = copy.deepcopy(spec.config)
        if vehicle_density is not None and "vehicles_density" in env_config:
            env_config["vehicles_density"] = vehicle_density

        env = gym.make(spec.env_id, config=env_config, render_mode=spec.render_mode)
        env.reset(seed=seed)
        return env


def init_env(
    env_name: str | None = None,
    vehicle_density: float | None = None,
    seed: int = 21,
    render_mode: str | None = None,
    stage: str = "self_play",
    scenario_name: str | None = None,
    env_config_overrides: dict | None = None,
):
    return EnvironmentFactory.create(
        env_name=env_name,
        vehicle_density=vehicle_density,
        seed=seed,
        render_mode=render_mode,
        stage=stage,
        scenario_name=scenario_name,
        env_config_overrides=env_config_overrides,
    )

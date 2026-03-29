from __future__ import annotations

import copy
from dataclasses import dataclass

import gymnasium as gym
import highway_env  # noqa: F401

from ..core.runtime_config import (
    get_active_scenario_name,
    get_environment_config,
    merge_config_dicts,
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
    @classmethod
    def create(
        cls,
        env_name: str | None = None,
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
        env = gym.make(spec.env_id, config=copy.deepcopy(spec.config), render_mode=spec.render_mode)
        env.reset(seed=seed)
        return env


def init_env(
    env_name: str | None = None,
    seed: int = 21,
    render_mode: str | None = None,
    stage: str = "self_play",
    scenario_name: str | None = None,
    env_config_overrides: dict | None = None,
):
    return EnvironmentFactory.create(
        env_name=env_name,
        seed=seed,
        render_mode=render_mode,
        stage=stage,
        scenario_name=scenario_name,
        env_config_overrides=env_config_overrides,
    )

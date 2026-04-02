from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym


def bootstrap_local_highway_env(source_root: str | Path) -> Path:
    local_highway_env_root = Path(source_root).expanduser().resolve() / "highway-env"
    if (
        local_highway_env_root.exists()
        and str(local_highway_env_root) not in sys.path
    ):
        # The adversarial environments rely on custom classes from the local fork.
        sys.path.insert(0, str(local_highway_env_root))

    import highway_env  # noqa: F401

    return local_highway_env_root


@dataclass(frozen=True)
class EnvironmentSpec:
    scenario_name: str
    env_id: str
    render_mode: str | None
    config: dict[str, Any]


class EnvironmentManager:
    def __init__(
        self,
        *,
        get_active_scenario_name: Callable[..., str],
        get_environment_config: Callable[..., dict[str, Any]],
        merge_config_dicts: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._get_active_scenario_name = get_active_scenario_name
        self._get_environment_config = get_environment_config
        self._merge_config_dicts = merge_config_dicts

    def build_env_spec(
        self,
        *,
        stage: str = "self_play",
        scenario_name: str | None = None,
        config_path: str | Path | None = None,
        env_name: str | None = None,
        render_mode: str | None = None,
        env_config_overrides: dict[str, Any] | None = None,
    ) -> EnvironmentSpec:
        resolved_scenario = scenario_name or self._get_active_scenario_name(
            config_path=config_path
        )
        environment = self._get_environment_config(
            stage=stage,
            scenario_name=resolved_scenario,
            config_path=config_path,
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
            env_config = self._merge_config_dicts(env_config, env_config_overrides)

        resolved_render_mode = (
            render_mode if render_mode is not None else environment.get("render_mode")
        )
        return EnvironmentSpec(
            scenario_name=resolved_scenario,
            env_id=env_id,
            render_mode=resolved_render_mode,
            config=env_config,
        )

    def create(
        self,
        *,
        env_name: str | None = None,
        seed: int = 21,
        render_mode: str | None = None,
        stage: str = "self_play",
        scenario_name: str | None = None,
        config_path: str | Path | None = None,
        env_config_overrides: dict[str, Any] | None = None,
    ):
        spec = self.build_env_spec(
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

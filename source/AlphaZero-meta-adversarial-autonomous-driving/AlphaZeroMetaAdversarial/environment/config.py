from __future__ import annotations

from pathlib import Path

from autonomous_driving_shared.alphazero_adversarial.environment.config import (
    EnvironmentManager,
    EnvironmentSpec,
    bootstrap_local_highway_env,
)

from ..core.runtime_config import (
    get_active_scenario_name,
    get_environment_config,
    merge_config_dicts,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SOURCE_ROOT = _REPO_ROOT / "source"
bootstrap_local_highway_env(_SOURCE_ROOT)

_ENVIRONMENT_MANAGER = EnvironmentManager(
    get_active_scenario_name=get_active_scenario_name,
    get_environment_config=get_environment_config,
    merge_config_dicts=merge_config_dicts,
)

build_env_spec = _ENVIRONMENT_MANAGER.build_env_spec


class EnvironmentFactory:
    @classmethod
    def create(
        cls,
        env_name: str | None = None,
        seed: int = 21,
        render_mode: str | None = None,
        stage: str = "self_play",
        scenario_name: str | None = None,
        config_path: str | Path | None = None,
        env_config_overrides: dict | None = None,
    ):
        return _ENVIRONMENT_MANAGER.create(
            env_name=env_name,
            seed=seed,
            render_mode=render_mode,
            stage=stage,
            scenario_name=scenario_name,
            config_path=config_path,
            env_config_overrides=env_config_overrides,
        )


def init_env(
    env_name: str | None = None,
    seed: int = 21,
    render_mode: str | None = None,
    stage: str = "self_play",
    scenario_name: str | None = None,
    config_path: str | Path | None = None,
    env_config_overrides: dict | None = None,
):
    return EnvironmentFactory.create(
        env_name=env_name,
        seed=seed,
        render_mode=render_mode,
        stage=stage,
        scenario_name=scenario_name,
        config_path=config_path,
        env_config_overrides=env_config_overrides,
    )


__all__ = [
    "EnvironmentFactory",
    "EnvironmentSpec",
    "build_env_spec",
    "init_env",
]

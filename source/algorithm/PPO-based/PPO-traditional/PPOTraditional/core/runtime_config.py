from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


class RuntimeConfigManager:
    def __init__(
        self,
        *,
        default_config_dir: str | os.PathLike[str],
        default_scenario_name: str,
        config_path_env_var: str,
        scenario_env_var: str,
        config_label: str,
        allowed_stages: tuple[str, ...] = ("train", "evaluation"),
    ) -> None:
        self.default_config_dir = Path(default_config_dir).expanduser().resolve()
        self.default_scenario_name = str(default_scenario_name)
        self.config_path_env_var = str(config_path_env_var)
        self.scenario_env_var = str(scenario_env_var)
        self.config_label = str(config_label)
        self.allowed_stages = tuple(str(stage) for stage in allowed_stages)

    def _scenario_config_candidates(
        self,
        scenario_name: str,
        config_dir: str | os.PathLike[str] | None = None,
    ) -> tuple[Path, ...]:
        directory = (
            Path(config_dir).expanduser().resolve()
            if config_dir is not None
            else self.default_config_dir
        )
        normalized_name = str(scenario_name).strip()
        return (
            directory / f"{normalized_name}.yaml",
            directory / f"{normalized_name}.yml",
        )

    def get_scenario_config_path(
        self,
        scenario_name: str,
        config_dir: str | os.PathLike[str] | None = None,
    ) -> Path:
        candidates = self._scenario_config_candidates(
            scenario_name,
            config_dir=config_dir,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        expected = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"No config file found for scenario {scenario_name!r}. Expected one of: {expected}"
        )

    def get_config_path(
        self,
        config_path: str | os.PathLike[str] | None = None,
    ) -> Path:
        raw_path = config_path or os.environ.get(self.config_path_env_var)
        if raw_path is not None:
            selected_path = Path(raw_path).expanduser().resolve()
        else:
            scenario_name = os.environ.get(
                self.scenario_env_var,
                self.default_scenario_name,
            )
            selected_path = self.get_scenario_config_path(scenario_name)
        if not selected_path.exists():
            raise FileNotFoundError(
                f"Configuration file does not exist: {selected_path}"
            )
        return selected_path

    def _load_serialized_config(self, path: Path) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        if yaml is not None:
            loaded = yaml.safe_load(text)
        else:
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "PyYAML is not installed, so the "
                    f"{self.config_label} config must use JSON-compatible YAML syntax."
                ) from exc
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Expected top-level mapping in {path}, got {type(loaded).__name__}."
            )
        return loaded

    @lru_cache(maxsize=None)
    def _load_runtime_config_cached(self, path_str: str) -> dict[str, Any]:
        return self._load_serialized_config(Path(path_str))

    def load_runtime_config(
        self,
        config_path: str | os.PathLike[str] | None = None,
        *,
        reload: bool = False,
    ) -> dict[str, Any]:
        path = self.get_config_path(config_path)
        if reload:
            self._load_runtime_config_cached.cache_clear()
        return copy.deepcopy(self._load_runtime_config_cached(str(path)))

    @staticmethod
    def merge_config_dicts(
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = RuntimeConfigManager.merge_config_dicts(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def _infer_scenario_name(
        self,
        loaded: dict[str, Any],
        path: Path | None = None,
    ) -> str:
        scenario_name = loaded.get("scenario_name")
        if scenario_name:
            return str(scenario_name)
        if path is not None:
            return path.stem
        return str(os.environ.get(self.scenario_env_var, self.default_scenario_name))

    def get_active_scenario_name(
        self,
        config_path: str | os.PathLike[str] | None = None,
        *,
        raw_config: dict[str, Any] | None = None,
    ) -> str:
        path = (
            self.get_config_path(config_path)
            if (config_path is not None or raw_config is None)
            else None
        )
        loaded = raw_config or self.load_runtime_config(path)
        override_name = os.environ.get(self.scenario_env_var)
        scenario_name = self._infer_scenario_name(loaded, path)
        if override_name and override_name != scenario_name:
            config_label = str(path) if path is not None else "the loaded config"
            raise ValueError(
                f"{self.scenario_env_var}={override_name!r} does not match "
                f"scenario {scenario_name!r} declared by {config_label}."
            )
        return scenario_name

    def get_scenario_config(
        self,
        scenario_name: str | None = None,
        config_path: str | os.PathLike[str] | None = None,
        *,
        raw_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = (
            self.get_config_path(config_path)
            if (config_path is not None or raw_config is None)
            else None
        )
        loaded = raw_config or self.load_runtime_config(path)
        declared_scenario_name = self._infer_scenario_name(loaded, path)
        expected_name = scenario_name or declared_scenario_name
        if expected_name != declared_scenario_name:
            config_label = str(path) if path is not None else "the loaded config"
            raise KeyError(
                f"Scenario {expected_name!r} does not match {config_label}, "
                f"which declares {declared_scenario_name!r}."
            )
        return copy.deepcopy(loaded)

    def get_environment_config(
        self,
        stage: str = "train",
        scenario_name: str | None = None,
        config_path: str | os.PathLike[str] | None = None,
        *,
        raw_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if stage not in self.allowed_stages:
            raise ValueError(
                f"Unknown stage {stage!r}. Expected one of {self.allowed_stages}."
            )
        scenario = self.get_scenario_config(
            scenario_name=scenario_name,
            config_path=config_path,
            raw_config=raw_config,
        )
        environment = scenario.get("environment")
        if not isinstance(environment, dict):
            raise ValueError("Scenario config must define an `environment` mapping.")
        return copy.deepcopy(environment)

    def get_stage_preset_config(
        self,
        stage: str,
        scenario_name: str | None = None,
        config_path: str | os.PathLike[str] | None = None,
        *,
        raw_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if stage not in self.allowed_stages:
            raise ValueError(
                f"Unknown stage {stage!r}. Expected one of {self.allowed_stages}."
            )
        scenario = self.get_scenario_config(
            scenario_name=scenario_name,
            config_path=config_path,
            raw_config=raw_config,
        )
        presets = scenario.get("presets")
        if not isinstance(presets, dict):
            raise ValueError("Scenario config must define a `presets` mapping.")
        preset = presets.get(stage)
        if not isinstance(preset, dict):
            raise KeyError(f"Scenario does not define a preset for stage {stage!r}.")
        return copy.deepcopy(preset)


_RUNTIME_CONFIG = RuntimeConfigManager(
    default_config_dir=Path(__file__).resolve().parents[2] / "configs",
    default_scenario_name="highway_ppo_traditional",
    config_path_env_var="PPO_TRADITIONAL_CONFIG_PATH",
    scenario_env_var="PPO_TRADITIONAL_SCENARIO",
    config_label="ppo-traditional",
    allowed_stages=("train", "evaluation"),
)

DEFAULT_CONFIG_DIR = _RUNTIME_CONFIG.default_config_dir
DEFAULT_SCENARIO_NAME = _RUNTIME_CONFIG.default_scenario_name
ALLOWED_STAGES = _RUNTIME_CONFIG.allowed_stages
get_scenario_config_path = _RUNTIME_CONFIG.get_scenario_config_path
get_config_path = _RUNTIME_CONFIG.get_config_path
load_runtime_config = _RUNTIME_CONFIG.load_runtime_config
merge_config_dicts = _RUNTIME_CONFIG.merge_config_dicts
get_active_scenario_name = _RUNTIME_CONFIG.get_active_scenario_name
get_scenario_config = _RUNTIME_CONFIG.get_scenario_config
get_environment_config = _RUNTIME_CONFIG.get_environment_config
get_stage_preset_config = _RUNTIME_CONFIG.get_stage_preset_config

__all__ = [
    "ALLOWED_STAGES",
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_SCENARIO_NAME",
    "RuntimeConfigManager",
    "get_active_scenario_name",
    "get_config_path",
    "get_environment_config",
    "get_scenario_config",
    "get_scenario_config_path",
    "get_stage_preset_config",
    "load_runtime_config",
    "merge_config_dicts",
]

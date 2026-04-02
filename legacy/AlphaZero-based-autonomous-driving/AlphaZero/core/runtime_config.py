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


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
DEFAULT_SCENARIO_NAME = "racetrack"
ALLOWED_STAGES = ("self_play", "inference", "evaluation")


def _scenario_config_candidates(
    scenario_name: str,
    config_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    directory = (
        Path(config_dir).expanduser().resolve()
        if config_dir is not None
        else DEFAULT_CONFIG_DIR.resolve()
    )
    normalized_name = str(scenario_name).strip()
    return (
        directory / f"{normalized_name}.yaml",
        directory / f"{normalized_name}.yml",
    )


def resolve_scenario_config_path(
    scenario_name: str,
    config_dir: str | os.PathLike[str] | None = None,
) -> Path:
    candidates = _scenario_config_candidates(scenario_name, config_dir=config_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    expected = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"No config file found for scenario {scenario_name!r}. Expected one of: {expected}"
    )


def resolve_config_path(config_path: str | os.PathLike[str] | None = None) -> Path:
    raw_path = config_path or os.environ.get("ALPHAZERO_CONFIG_PATH")
    if raw_path is not None:
        resolved = Path(raw_path).expanduser().resolve()
    else:
        scenario_name = os.environ.get("ALPHAZERO_SCENARIO", DEFAULT_SCENARIO_NAME)
        resolved = resolve_scenario_config_path(scenario_name)
    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {resolved}")
    return resolved


def _load_serialized_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        loaded = yaml.safe_load(text)
    else:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "PyYAML is not installed, so the scenario config must use JSON-compatible YAML syntax. "
                "Install `pyyaml` to enable general YAML parsing."
            ) from exc

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected top-level mapping in {path}, got {type(loaded).__name__}."
        )
    return loaded


@lru_cache(maxsize=None)
def _load_runtime_config_cached(path_str: str) -> dict[str, Any]:
    return _load_serialized_config(Path(path_str))


def load_runtime_config(
    config_path: str | os.PathLike[str] | None = None,
    *,
    reload: bool = False,
) -> dict[str, Any]:
    path = resolve_config_path(config_path)
    if reload:
        _load_runtime_config_cached.cache_clear()
    return copy.deepcopy(_load_runtime_config_cached(str(path)))


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _infer_scenario_name(
    loaded: dict[str, Any],
    path: Path | None = None,
) -> str:
    scenario_name = loaded.get("scenario_name")
    if scenario_name:
        return str(scenario_name)
    if path is not None:
        return path.stem
    return str(os.environ.get("ALPHAZERO_SCENARIO", DEFAULT_SCENARIO_NAME))


def get_available_scenarios(
    config_path: str | os.PathLike[str] | None = None,
    *,
    raw_config: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    if raw_config is not None:
        return (_infer_scenario_name(raw_config),)

    if config_path is not None:
        candidate = Path(config_path).expanduser().resolve()
        if candidate.is_file():
            loaded = load_runtime_config(candidate)
            return (_infer_scenario_name(loaded, candidate),)
        search_dir = candidate
    else:
        search_dir = DEFAULT_CONFIG_DIR.resolve()

    if not search_dir.exists():
        raise FileNotFoundError(f"Config directory does not exist: {search_dir}")

    scenario_names = sorted(
        {
            path.stem
            for pattern in ("*.yaml", "*.yml")
            for path in search_dir.glob(pattern)
            if path.is_file()
        }
    )
    if not scenario_names:
        raise ValueError(f"No scenario config files were found under {search_dir}.")
    return tuple(scenario_names)


def get_active_scenario_name(
    config_path: str | os.PathLike[str] | None = None,
    *,
    raw_config: dict[str, Any] | None = None,
) -> str:
    path = resolve_config_path(config_path) if (config_path is not None or raw_config is None) else None
    loaded = raw_config or load_runtime_config(path)
    override_name = os.environ.get("ALPHAZERO_SCENARIO")
    scenario_name = _infer_scenario_name(loaded, path)
    if override_name and override_name != scenario_name:
        config_label = str(path) if path is not None else "the loaded config"
        raise ValueError(
            f"ALPHAZERO_SCENARIO={override_name!r} does not match scenario "
            f"{scenario_name!r} declared by {config_label}."
        )
    return scenario_name


def get_scenario_config(
    scenario_name: str | None = None,
    config_path: str | os.PathLike[str] | None = None,
    *,
    raw_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = resolve_config_path(config_path) if (config_path is not None or raw_config is None) else None
    loaded = raw_config or load_runtime_config(path)
    resolved_name = _infer_scenario_name(loaded, path)
    expected_name = scenario_name or resolved_name
    if expected_name != resolved_name:
        config_label = str(path) if path is not None else "the loaded config"
        raise KeyError(
            f"Scenario {expected_name!r} does not match {config_label}, "
            f"which declares {resolved_name!r}."
        )
    return copy.deepcopy(loaded)


def get_stage_preset_config(
    stage: str,
    scenario_name: str | None = None,
    config_path: str | os.PathLike[str] | None = None,
    *,
    raw_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Expected one of {ALLOWED_STAGES}.")
    path = resolve_config_path(config_path) if (config_path is not None or raw_config is None) else None
    loaded = raw_config or load_runtime_config(path)
    resolved_scenario = scenario_name or _infer_scenario_name(loaded, path)

    scenario = get_scenario_config(
        scenario_name=resolved_scenario,
        config_path=path,
        raw_config=loaded,
    )
    presets = scenario.get("presets")
    if not isinstance(presets, dict):
        raise ValueError(
            f"Scenario config for {resolved_scenario!r} must define a `presets` mapping."
        )
    preset = presets.get(stage)
    if not isinstance(preset, dict):
        raise KeyError(
            f"Scenario {resolved_scenario!r} does not define a preset for stage {stage!r}."
        )
    return copy.deepcopy(preset)


def get_environment_config(
    stage: str = "self_play",
    scenario_name: str | None = None,
    config_path: str | os.PathLike[str] | None = None,
    *,
    raw_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Expected one of {ALLOWED_STAGES}.")
    path = resolve_config_path(config_path) if (config_path is not None or raw_config is None) else None
    loaded = raw_config or load_runtime_config(path)
    resolved_scenario = scenario_name or _infer_scenario_name(loaded, path)

    scenario = get_scenario_config(
        scenario_name=resolved_scenario,
        config_path=path,
        raw_config=loaded,
    )
    environment = scenario.get("environment")
    if not isinstance(environment, dict):
        raise ValueError(
            f"Scenario config for {resolved_scenario!r} must define an `environment` mapping."
        )

    stage_overrides = scenario.get("environment_overrides", {})
    if stage_overrides and not isinstance(stage_overrides, dict):
        raise ValueError("`environment_overrides` must be a mapping when provided.")

    raw_stage_override = (
        stage_overrides.get(stage, {}) if isinstance(stage_overrides, dict) else {}
    )
    if raw_stage_override and not isinstance(raw_stage_override, dict):
        raise ValueError(
            f"`environment_overrides.{stage}` must be a mapping when provided."
        )

    return _deep_merge_dicts(environment, raw_stage_override)


def merge_config_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    return _deep_merge_dicts(base, override)

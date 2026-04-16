from __future__ import annotations

from pathlib import Path

from autonomous_driving_shared.alphazero_adversarial.core.runtime_config import (
    RuntimeConfigManager,
)


_RUNTIME_CONFIG = RuntimeConfigManager(
    default_config_dir=Path(__file__).resolve().parents[2] / "configs",
    default_scenario_name="highway_meta_adversarial",
    config_path_env_var="ALPHAZERO_META_ADVERSARIAL_CONFIG_PATH",
    scenario_env_var="ALPHAZERO_META_ADVERSARIAL_SCENARIO",
    config_label="meta-adversarial",
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

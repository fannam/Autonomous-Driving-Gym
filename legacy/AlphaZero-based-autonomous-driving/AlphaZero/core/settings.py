from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from core.runtime_config import (
        get_active_scenario_name,
        get_environment_config,
        get_stage_preset_config,
        load_runtime_config,
        resolve_config_path,
    )
except ModuleNotFoundError as exc:
    if exc.name != "core":
        raise
    from .runtime_config import (
        get_active_scenario_name,
        get_environment_config,
        get_stage_preset_config,
        load_runtime_config,
        resolve_config_path,
    )


def compute_grid_shape(grid_size, grid_step):
    grid_size_arr = np.asarray(grid_size, dtype=np.float32)
    grid_step_arr = np.asarray(grid_step, dtype=np.float32)
    grid_shape = np.floor((grid_size_arr[:, 1] - grid_size_arr[:, 0]) / grid_step_arr)
    return tuple(int(axis_cells) for axis_cells in grid_shape)


CONFIG_PATH = resolve_config_path()
RAW_CONFIG = load_runtime_config(CONFIG_PATH)
ACTIVE_SCENARIO = get_active_scenario_name(raw_config=RAW_CONFIG)


def _resolve_observation_grid_metadata(
    *,
    stage: str,
    scenario_name: str | None = None,
    raw_config: dict | None = None,
) -> tuple[list[list[float]], list[float]]:
    environment = get_environment_config(
        stage=stage,
        scenario_name=scenario_name,
        raw_config=raw_config,
    )
    env_config = environment.get("config", {})
    observation = env_config.get("observation", {})
    grid_size = observation.get("grid_size", [[-50, 50], [-12, 12]])
    grid_step = observation.get("grid_step", [1.0, 1.0])
    return grid_size, grid_step


DEFAULT_GRID_SIZE, DEFAULT_GRID_STEP = _resolve_observation_grid_metadata(
    stage="self_play",
    scenario_name=ACTIVE_SCENARIO,
    raw_config=RAW_CONFIG,
)
DEFAULT_GRID_SHAPE = compute_grid_shape(DEFAULT_GRID_SIZE, DEFAULT_GRID_STEP)
DEFAULT_EGO_POSITION = (
    DEFAULT_GRID_SHAPE[0] // 2,
    DEFAULT_GRID_SHAPE[1] // 2,
)


@dataclass(frozen=True)
class StackConfig:
    grid_size: tuple[int, int] = field(default_factory=lambda: DEFAULT_GRID_SHAPE)
    ego_position: tuple[int, int] | None = field(
        default_factory=lambda: DEFAULT_EGO_POSITION
    )
    history_length: int = 5
    append_raw_ego_speed_plane: bool = False
    mark_ego_in_init: bool = True
    presence_feature_index: int = 0
    lane_feature_index: int = 1
    presence_feature_name: str = "presence"
    lane_feature_name: str = "on_lane"
    lane_feature_fallback_name: str = "on_road"

    @classmethod
    def from_dict(
        cls,
        raw_config: dict | None = None,
        *,
        default_grid_size: tuple[int, int] | None = None,
        default_ego_position: tuple[int, int] | None = None,
    ) -> "StackConfig":
        data = dict(raw_config or {})
        if default_grid_size is None:
            default_grid_size = DEFAULT_GRID_SHAPE
        if default_ego_position is None:
            default_ego_position = DEFAULT_EGO_POSITION
        if "grid_shape" in data and "grid_size" not in data:
            data["grid_size"] = data.pop("grid_shape")

        if "grid_size" not in data or data["grid_size"] is None:
            data["grid_size"] = default_grid_size
        else:
            data["grid_size"] = tuple(int(value) for value in data["grid_size"])

        if "ego_position" not in data:
            data["ego_position"] = default_ego_position
        elif data["ego_position"] is not None:
            data["ego_position"] = tuple(int(value) for value in data["ego_position"])

        if data.get("ego_position") is None:
            data["ego_position"] = (
                data["grid_size"][0] // 2,
                data["grid_size"][1] // 2,
            )

        return cls(**data)

    @property
    def plane_count(self) -> int:
        # history + lane + speed_norm + speed_delta + speed_min + speed_max
        # (+ optional raw ego-speed plane)
        return self.history_length + 5 + int(self.append_raw_ego_speed_plane)

    @property
    def network_input_shape(self) -> tuple[int, int, int]:
        width, height = self.grid_size
        return (width, height, self.plane_count)


@dataclass(frozen=True)
class AlphaZeroConfig:
    stack: StackConfig = field(
        default_factory=lambda: StackConfig(
            grid_size=DEFAULT_GRID_SHAPE,
            ego_position=DEFAULT_EGO_POSITION,
        )
    )
    n_actions: int = 25
    n_residual_layers: int = 10
    c_puct: float = 2.5
    n_simulations: int = 5
    temperature: float = 1.0
    temperature_drop_step: int | None = None
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    model_path: str = "alphazero_model (19).pth"

    @classmethod
    def from_dict(
        cls,
        raw_config: dict | None = None,
        *,
        default_stack_config: StackConfig | None = None,
    ) -> "AlphaZeroConfig":
        data = dict(raw_config or {})
        stack = StackConfig.from_dict(
            data.pop("stack", {}),
            default_grid_size=(
                DEFAULT_GRID_SHAPE
                if default_stack_config is None
                else default_stack_config.grid_size
            ),
            default_ego_position=(
                DEFAULT_EGO_POSITION
                if default_stack_config is None
                else default_stack_config.ego_position
            ),
        )
        data["stack"] = stack
        return cls(**data)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self.stack.network_input_shape


def load_stage_config(
    stage: str,
    *,
    scenario_name: str | None = None,
    raw_config: dict | None = None,
) -> AlphaZeroConfig:
    loaded = RAW_CONFIG if raw_config is None else raw_config
    resolved_scenario = scenario_name or get_active_scenario_name(raw_config=loaded)
    grid_size, grid_step = _resolve_observation_grid_metadata(
        stage=stage,
        scenario_name=resolved_scenario,
        raw_config=loaded,
    )
    grid_shape = compute_grid_shape(grid_size, grid_step)
    ego_position = (grid_shape[0] // 2, grid_shape[1] // 2)
    preset = get_stage_preset_config(
        stage=stage,
        scenario_name=resolved_scenario,
        raw_config=loaded,
    )
    return AlphaZeroConfig.from_dict(
        preset,
        default_stack_config=StackConfig(
            grid_size=grid_shape,
            ego_position=ego_position,
        ),
    )


def reload_settings() -> tuple[AlphaZeroConfig, AlphaZeroConfig, AlphaZeroConfig]:
    global CONFIG_PATH, RAW_CONFIG, ACTIVE_SCENARIO
    global DEFAULT_GRID_SIZE, DEFAULT_GRID_STEP, DEFAULT_GRID_SHAPE, DEFAULT_EGO_POSITION
    global SELF_PLAY_CONFIG, INFERENCE_CONFIG, EVALUATION_CONFIG

    CONFIG_PATH = resolve_config_path()
    RAW_CONFIG = load_runtime_config(CONFIG_PATH, reload=True)
    ACTIVE_SCENARIO = get_active_scenario_name(raw_config=RAW_CONFIG)
    DEFAULT_GRID_SIZE, DEFAULT_GRID_STEP = _resolve_observation_grid_metadata(
        stage="self_play",
        scenario_name=ACTIVE_SCENARIO,
        raw_config=RAW_CONFIG,
    )
    DEFAULT_GRID_SHAPE = compute_grid_shape(DEFAULT_GRID_SIZE, DEFAULT_GRID_STEP)
    DEFAULT_EGO_POSITION = (
        DEFAULT_GRID_SHAPE[0] // 2,
        DEFAULT_GRID_SHAPE[1] // 2,
    )
    SELF_PLAY_CONFIG = load_stage_config("self_play", raw_config=RAW_CONFIG)
    INFERENCE_CONFIG = load_stage_config("inference", raw_config=RAW_CONFIG)
    EVALUATION_CONFIG = load_stage_config("evaluation", raw_config=RAW_CONFIG)
    return SELF_PLAY_CONFIG, INFERENCE_CONFIG, EVALUATION_CONFIG


SELF_PLAY_CONFIG = load_stage_config("self_play", raw_config=RAW_CONFIG)
INFERENCE_CONFIG = load_stage_config("inference", raw_config=RAW_CONFIG)
EVALUATION_CONFIG = load_stage_config("evaluation", raw_config=RAW_CONFIG)

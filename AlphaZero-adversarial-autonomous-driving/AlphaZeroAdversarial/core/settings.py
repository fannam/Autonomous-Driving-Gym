from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .runtime_config import (
    get_active_scenario_name,
    get_environment_config,
    get_stage_preset_config,
    load_runtime_config,
    resolve_config_path,
)


def compute_grid_shape(grid_size, grid_step) -> tuple[int, int]:
    grid_size_arr = np.asarray(grid_size, dtype=np.float32)
    grid_step_arr = np.asarray(grid_step, dtype=np.float32)
    grid_shape = np.floor((grid_size_arr[:, 1] - grid_size_arr[:, 0]) / grid_step_arr)
    return tuple(int(axis_cells) for axis_cells in grid_shape)


CONFIG_PATH = resolve_config_path()
RAW_CONFIG = load_runtime_config(CONFIG_PATH)
ACTIVE_SCENARIO = get_active_scenario_name(raw_config=RAW_CONFIG)


def _resolve_observation_metadata(
    *,
    stage: str,
    scenario_name: str | None = None,
    raw_config: dict | None = None,
) -> tuple[list[list[float]], list[float], tuple[str, ...]]:
    environment = get_environment_config(
        stage=stage,
        scenario_name=scenario_name,
        raw_config=raw_config,
    )
    env_config = environment.get("config", {})
    observation = env_config.get("observation", {})
    if observation.get("type") == "MultiAgentObservation":
        observation = observation.get("observation_config", {})
    grid_size = observation.get("grid_size", [[-50, 50], [-12, 12]])
    grid_step = observation.get("grid_step", [1.0, 1.0])
    features = tuple(observation.get("features", ["on_lane", "on_road"]))
    return grid_size, grid_step, features


def _resolve_action_metadata(
    *,
    stage: str,
    scenario_name: str | None = None,
    raw_config: dict | None = None,
) -> tuple[int, int]:
    environment = get_environment_config(
        stage=stage,
        scenario_name=scenario_name,
        raw_config=raw_config,
    )
    env_config = environment.get("config", {})
    action = env_config.get("action", {})
    if action.get("type") == "MultiAgentAction":
        action = action.get("action_config", {})

    actions_per_axis = int(action.get("actions_per_axis", 5))
    longitudinal = bool(action.get("longitudinal", True))
    lateral = bool(action.get("lateral", True))
    return (
        actions_per_axis if longitudinal else 1,
        actions_per_axis if lateral else 1,
    )


DEFAULT_GRID_EXTENT, DEFAULT_GRID_STEP, DEFAULT_STATIC_FEATURES = _resolve_observation_metadata(
    stage="self_play",
    scenario_name=ACTIVE_SCENARIO,
    raw_config=RAW_CONFIG,
)
DEFAULT_GRID_SHAPE = compute_grid_shape(DEFAULT_GRID_EXTENT, DEFAULT_GRID_STEP)
DEFAULT_ACTION_AXIS_0, DEFAULT_ACTION_AXIS_1 = _resolve_action_metadata(
    stage="self_play",
    scenario_name=ACTIVE_SCENARIO,
    raw_config=RAW_CONFIG,
)


@dataclass(frozen=True)
class PerspectiveTensorConfig:
    grid_shape: tuple[int, int] = field(default_factory=lambda: DEFAULT_GRID_SHAPE)
    grid_extent: tuple[tuple[float, float], tuple[float, float]] = field(
        default_factory=lambda: tuple(tuple(float(value) for value in axis) for axis in DEFAULT_GRID_EXTENT)
    )
    grid_step: tuple[float, float] = field(
        default_factory=lambda: tuple(float(value) for value in DEFAULT_GRID_STEP)
    )
    history_length: int = 5
    static_feature_names: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_STATIC_FEATURES
    )
    include_self_speed_plane: bool = True
    include_heading_planes: bool = True
    include_progress_plane: bool = True
    flip_npc_perspective: bool = True

    @classmethod
    def from_dict(
        cls,
        raw_config: dict | None = None,
        *,
        default_grid_shape: tuple[int, int] | None = None,
        default_grid_extent: tuple[tuple[float, float], tuple[float, float]] | None = None,
        default_grid_step: tuple[float, float] | None = None,
        default_static_feature_names: tuple[str, ...] | None = None,
    ) -> "PerspectiveTensorConfig":
        data = dict(raw_config or {})
        if default_grid_shape is None:
            default_grid_shape = DEFAULT_GRID_SHAPE
        if default_grid_extent is None:
            default_grid_extent = tuple(
                tuple(float(value) for value in axis) for axis in DEFAULT_GRID_EXTENT
            )
        if default_grid_step is None:
            default_grid_step = tuple(float(value) for value in DEFAULT_GRID_STEP)
        if default_static_feature_names is None:
            default_static_feature_names = DEFAULT_STATIC_FEATURES

        data["grid_shape"] = tuple(
            int(value) for value in data.get("grid_shape", default_grid_shape)
        )
        data["grid_extent"] = tuple(
            tuple(float(value) for value in axis)
            for axis in data.get("grid_extent", default_grid_extent)
        )
        data["grid_step"] = tuple(
            float(value) for value in data.get("grid_step", default_grid_step)
        )
        data["static_feature_names"] = tuple(
            str(value)
            for value in data.get("static_feature_names", default_static_feature_names)
        )
        return cls(**data)

    @property
    def scalar_plane_count(self) -> int:
        return (
            int(self.include_self_speed_plane)
            + 2 * int(self.include_heading_planes)
            + int(self.include_progress_plane)
        )

    @property
    def k_channels(self) -> int:
        return len(self.static_feature_names) + self.scalar_plane_count

    @property
    def plane_count(self) -> int:
        return 2 * self.history_length + self.k_channels

    @property
    def network_input_shape(self) -> tuple[int, int, int]:
        width, height = self.grid_shape
        return (width, height, self.plane_count)


@dataclass(frozen=True)
class ZeroSumConfig:
    minimum_safe_speed: float = 5.0

    @classmethod
    def from_dict(cls, raw_config: dict | None = None) -> "ZeroSumConfig":
        return cls(**dict(raw_config or {}))


@dataclass(frozen=True)
class AdversarialAlphaZeroConfig:
    tensor: PerspectiveTensorConfig = field(default_factory=PerspectiveTensorConfig)
    zero_sum: ZeroSumConfig = field(default_factory=ZeroSumConfig)
    n_actions: int = 25
    n_action_axis_0: int = DEFAULT_ACTION_AXIS_0
    n_action_axis_1: int = DEFAULT_ACTION_AXIS_1
    n_residual_layers: int = 10
    network_channels: int = 256
    network_dropout_p: float = 0.1
    c_puct: float = 2.5
    n_simulations: int = 24
    temperature: float = 1.0
    temperature_drop_step: int | None = None
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    relative_pruning_gamma: float | None = 0.1
    max_expand_actions_per_agent: int | None = 6
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    replay_buffer_size: int = 4000
    use_policy_target_smoothing: bool = False
    policy_target_smoothing_sigma: float = 1.0
    warmup_episodes: int = 0
    warmup_opponent_policy: str | None = None
    warmup_collect_opponent_samples: bool = True
    model_path: str = "models/alphazero_adversarial_racetrack.pth"

    def __post_init__(self) -> None:
        if int(self.n_action_axis_0) <= 0 or int(self.n_action_axis_1) <= 0:
            raise ValueError("Action-axis sizes must be positive integers.")

        expected_n_actions = int(self.n_action_axis_0) * int(self.n_action_axis_1)
        if int(self.n_actions) != expected_n_actions:
            raise ValueError(
                "Expected n_actions to match the factorized action shape, "
                f"got n_actions={self.n_actions} and "
                f"shape=({self.n_action_axis_0}, {self.n_action_axis_1})."
            )

        if (
            self.relative_pruning_gamma is not None
            and float(self.relative_pruning_gamma) < 0.0
        ):
            raise ValueError("relative_pruning_gamma must be non-negative or None.")

        if (
            bool(self.use_policy_target_smoothing)
            and float(self.policy_target_smoothing_sigma) <= 0.0
        ):
            raise ValueError(
                "policy_target_smoothing_sigma must be positive when smoothing is enabled."
            )

    @classmethod
    def from_dict(
        cls,
        raw_config: dict | None = None,
        *,
        default_tensor_config: PerspectiveTensorConfig | None = None,
        default_n_action_axis_0: int | None = None,
        default_n_action_axis_1: int | None = None,
    ) -> "AdversarialAlphaZeroConfig":
        data = dict(raw_config or {})
        tensor = PerspectiveTensorConfig.from_dict(
            data.pop("tensor", {}),
            default_grid_shape=(
                DEFAULT_GRID_SHAPE
                if default_tensor_config is None
                else default_tensor_config.grid_shape
            ),
            default_grid_extent=(
                tuple(tuple(float(value) for value in axis) for axis in DEFAULT_GRID_EXTENT)
                if default_tensor_config is None
                else default_tensor_config.grid_extent
            ),
            default_grid_step=(
                tuple(float(value) for value in DEFAULT_GRID_STEP)
                if default_tensor_config is None
                else default_tensor_config.grid_step
            ),
            default_static_feature_names=(
                DEFAULT_STATIC_FEATURES
                if default_tensor_config is None
                else default_tensor_config.static_feature_names
            ),
        )
        zero_sum = ZeroSumConfig.from_dict(data.pop("zero_sum", {}))
        if default_n_action_axis_0 is None:
            default_n_action_axis_0 = DEFAULT_ACTION_AXIS_0
        if default_n_action_axis_1 is None:
            default_n_action_axis_1 = DEFAULT_ACTION_AXIS_1

        data["n_action_axis_0"] = int(
            data.get("n_action_axis_0", default_n_action_axis_0)
        )
        data["n_action_axis_1"] = int(
            data.get("n_action_axis_1", default_n_action_axis_1)
        )
        data["n_actions"] = int(
            data.get(
                "n_actions",
                int(data["n_action_axis_0"]) * int(data["n_action_axis_1"]),
            )
        )
        data["tensor"] = tensor
        data["zero_sum"] = zero_sum
        return cls(**data)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self.tensor.network_input_shape

    @property
    def action_shape(self) -> tuple[int, int]:
        return int(self.n_action_axis_0), int(self.n_action_axis_1)


def load_stage_config(
    stage: str,
    *,
    scenario_name: str | None = None,
    raw_config: dict | None = None,
) -> AdversarialAlphaZeroConfig:
    loaded = RAW_CONFIG if raw_config is None else raw_config
    resolved_scenario = scenario_name or get_active_scenario_name(raw_config=loaded)
    grid_extent, grid_step, static_features = _resolve_observation_metadata(
        stage=stage,
        scenario_name=resolved_scenario,
        raw_config=loaded,
    )
    action_axis_0, action_axis_1 = _resolve_action_metadata(
        stage=stage,
        scenario_name=resolved_scenario,
        raw_config=loaded,
    )
    preset = get_stage_preset_config(
        stage=stage,
        scenario_name=resolved_scenario,
        raw_config=loaded,
    )
    return AdversarialAlphaZeroConfig.from_dict(
        preset,
        default_tensor_config=PerspectiveTensorConfig(
            grid_shape=compute_grid_shape(grid_extent, grid_step),
            grid_extent=tuple(tuple(float(value) for value in axis) for axis in grid_extent),
            grid_step=tuple(float(value) for value in grid_step),
            static_feature_names=tuple(str(value) for value in static_features),
        ),
        default_n_action_axis_0=action_axis_0,
        default_n_action_axis_1=action_axis_1,
    )


def reload_settings() -> tuple[
    AdversarialAlphaZeroConfig,
    AdversarialAlphaZeroConfig,
    AdversarialAlphaZeroConfig,
]:
    global CONFIG_PATH, RAW_CONFIG, ACTIVE_SCENARIO
    global DEFAULT_GRID_EXTENT, DEFAULT_GRID_STEP, DEFAULT_STATIC_FEATURES, DEFAULT_GRID_SHAPE
    global DEFAULT_ACTION_AXIS_0, DEFAULT_ACTION_AXIS_1
    global SELF_PLAY_CONFIG, INFERENCE_CONFIG, EVALUATION_CONFIG

    CONFIG_PATH = resolve_config_path()
    RAW_CONFIG = load_runtime_config(CONFIG_PATH, reload=True)
    ACTIVE_SCENARIO = get_active_scenario_name(raw_config=RAW_CONFIG)
    (
        DEFAULT_GRID_EXTENT,
        DEFAULT_GRID_STEP,
        DEFAULT_STATIC_FEATURES,
    ) = _resolve_observation_metadata(
        stage="self_play",
        scenario_name=ACTIVE_SCENARIO,
        raw_config=RAW_CONFIG,
    )
    DEFAULT_GRID_SHAPE = compute_grid_shape(DEFAULT_GRID_EXTENT, DEFAULT_GRID_STEP)
    DEFAULT_ACTION_AXIS_0, DEFAULT_ACTION_AXIS_1 = _resolve_action_metadata(
        stage="self_play",
        scenario_name=ACTIVE_SCENARIO,
        raw_config=RAW_CONFIG,
    )
    SELF_PLAY_CONFIG = load_stage_config("self_play", raw_config=RAW_CONFIG)
    INFERENCE_CONFIG = load_stage_config("inference", raw_config=RAW_CONFIG)
    EVALUATION_CONFIG = load_stage_config("evaluation", raw_config=RAW_CONFIG)
    return SELF_PLAY_CONFIG, INFERENCE_CONFIG, EVALUATION_CONFIG


SELF_PLAY_CONFIG = load_stage_config("self_play", raw_config=RAW_CONFIG)
INFERENCE_CONFIG = load_stage_config("inference", raw_config=RAW_CONFIG)
EVALUATION_CONFIG = load_stage_config("evaluation", raw_config=RAW_CONFIG)

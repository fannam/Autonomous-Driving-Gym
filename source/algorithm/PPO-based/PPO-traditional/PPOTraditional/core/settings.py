from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .runtime_config import (
    get_active_scenario_name,
    get_config_path,
    get_environment_config,
    get_stage_preset_config,
    load_runtime_config,
)


def compute_grid_shape(grid_size, grid_step) -> tuple[int, int]:
    grid_size_arr = np.asarray(grid_size, dtype=np.float32)
    grid_step_arr = np.asarray(grid_step, dtype=np.float32)
    grid_shape = np.floor((grid_size_arr[:, 1] - grid_size_arr[:, 0]) / grid_step_arr)
    return tuple(int(axis_cells) for axis_cells in grid_shape)


CONFIG_PATH = get_config_path()
RAW_CONFIG = load_runtime_config(CONFIG_PATH)
ACTIVE_SCENARIO = get_active_scenario_name(raw_config=RAW_CONFIG)


def _resolve_observation_config(environment_config: dict) -> dict:
    env_config = environment_config.get("config", {})
    observation = env_config.get("observation", {})
    if observation.get("type") == "MultiAgentObservation":
        observation = observation.get("observation_config", {})
    if not isinstance(observation, dict):
        raise ValueError("Environment config must define a valid observation mapping.")
    return observation


def _get_observation_metadata(
    *,
    stage: str,
    scenario_name: str | None = None,
    raw_config: dict | None = None,
) -> tuple[list[list[float]], list[float], tuple[str, ...], tuple[int, int, int]]:
    environment = get_environment_config(
        stage=stage,
        scenario_name=scenario_name,
        raw_config=raw_config,
    )
    observation = _resolve_observation_config(environment)
    grid_size = observation.get("grid_size", [[-50, 50], [-12, 12]])
    grid_step = observation.get("grid_step", [1.0, 1.0])
    features = tuple(observation.get("features", ["presence", "on_lane", "on_road"]))
    grid_shape = compute_grid_shape(grid_size, grid_step)
    observation_shape = (len(features), *grid_shape)
    return grid_size, grid_step, features, observation_shape


def _get_action_count(
    *,
    stage: str,
    scenario_name: str | None = None,
    raw_config: dict | None = None,
) -> int:
    environment = get_environment_config(
        stage=stage,
        scenario_name=scenario_name,
        raw_config=raw_config,
    )
    env_config = environment.get("config", {})
    action = env_config.get("action", {})
    if action.get("type") == "MultiAgentAction":
        action = action.get("action_config", {})

    action_type = str(action.get("type", "DiscreteMetaAction"))
    longitudinal = bool(action.get("longitudinal", True))
    lateral = bool(action.get("lateral", True))

    if action_type == "DiscreteMetaAction":
        if longitudinal and lateral:
            return 5
        if longitudinal or lateral:
            return 3
        raise ValueError(
            "DiscreteMetaAction must enable longitudinal and/or lateral controls."
        )

    if action_type == "DiscreteAction":
        actions_per_axis = int(action.get("actions_per_axis", 5))
        axis_count = int(longitudinal) + int(lateral)
        return int(actions_per_axis ** max(1, axis_count))

    raise ValueError(f"Unsupported action type {action_type!r} for PPO-traditional settings.")


(
    DEFAULT_GRID_EXTENT,
    DEFAULT_GRID_STEP,
    DEFAULT_FEATURES,
    DEFAULT_OBSERVATION_SHAPE,
) = _get_observation_metadata(
    stage="train",
    scenario_name=ACTIVE_SCENARIO,
    raw_config=RAW_CONFIG,
)
DEFAULT_N_ACTIONS = _get_action_count(
    stage="train",
    scenario_name=ACTIVE_SCENARIO,
    raw_config=RAW_CONFIG,
)


@dataclass(frozen=True)
class RewardConfig:
    speed_weight: float = 1.0
    right_lane_weight: float = 0.1
    collision_penalty: float = 5.0
    offroad_penalty: float = 5.0

    @classmethod
    def from_dict(cls, raw_config: dict | None = None) -> "RewardConfig":
        return cls(**dict(raw_config or {}))


@dataclass(frozen=True)
class NetworkConfig:
    stem_channels: int = 32
    downsample_channels: int = 64
    latent_dim: int = 256
    residual_blocks: int = 4
    dropout_p: float = 0.1

    @classmethod
    def from_dict(cls, raw_config: dict | None = None) -> "NetworkConfig":
        return cls(**dict(raw_config or {}))


@dataclass(frozen=True)
class RolloutConfig:
    n_envs: int = 8
    steps_per_env: int = 64
    max_steps: int | None = None

    @classmethod
    def from_dict(cls, raw_config: dict | None = None) -> "RolloutConfig":
        return cls(**dict(raw_config or {}))


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 2
    minibatch_size: int = 128
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02

    @classmethod
    def from_dict(cls, raw_config: dict | None = None) -> "PPOConfig":
        return cls(**dict(raw_config or {}))


@dataclass(frozen=True)
class LoggingConfig:
    metrics_path: str = "outputs/ppo_traditional_metrics.jsonl"

    @classmethod
    def from_dict(cls, raw_config: dict | None = None) -> "LoggingConfig":
        return cls(**dict(raw_config or {}))


@dataclass(frozen=True)
class PPOTraditionalConfig:
    observation_shape: tuple[int, int, int] = field(
        default_factory=lambda: DEFAULT_OBSERVATION_SHAPE
    )
    observation_features: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_FEATURES
    )
    n_actions: int = DEFAULT_N_ACTIONS
    reward: RewardConfig = field(default_factory=RewardConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model_path: str = "models/ppo_traditional_highway.pt"

    def __post_init__(self) -> None:
        if len(self.observation_shape) != 3:
            raise ValueError(
                f"Expected observation_shape=(channels, width, height), got {self.observation_shape!r}."
            )
        if int(self.n_actions) <= 0:
            raise ValueError("n_actions must be positive.")
        if int(self.network.residual_blocks) <= 0:
            raise ValueError("network.residual_blocks must be positive.")
        if int(self.rollout.n_envs) <= 0:
            raise ValueError("rollout.n_envs must be positive.")
        if int(self.rollout.steps_per_env) <= 0:
            raise ValueError("rollout.steps_per_env must be positive.")
        if int(self.ppo.ppo_epochs) <= 0:
            raise ValueError("ppo.ppo_epochs must be positive.")
        if int(self.ppo.minibatch_size) <= 0:
            raise ValueError("ppo.minibatch_size must be positive.")
        if not (0.0 < float(self.ppo.gamma) <= 1.0):
            raise ValueError("ppo.gamma must be in (0, 1].")
        if not (0.0 <= float(self.ppo.gae_lambda) <= 1.0):
            raise ValueError("ppo.gae_lambda must be in [0, 1].")

    @classmethod
    def from_dict(
        cls,
        raw_config: dict | None = None,
        *,
        default_observation_shape: tuple[int, int, int] | None = None,
        default_observation_features: tuple[str, ...] | None = None,
        default_n_actions: int | None = None,
    ) -> "PPOTraditionalConfig":
        data = dict(raw_config or {})
        if default_observation_shape is None:
            default_observation_shape = DEFAULT_OBSERVATION_SHAPE
        if default_observation_features is None:
            default_observation_features = DEFAULT_FEATURES
        if default_n_actions is None:
            default_n_actions = DEFAULT_N_ACTIONS

        data["observation_shape"] = tuple(
            int(value)
            for value in data.get("observation_shape", default_observation_shape)
        )
        data["observation_features"] = tuple(
            str(value)
            for value in data.get("observation_features", default_observation_features)
        )
        data["n_actions"] = int(data.get("n_actions", default_n_actions))
        data.pop("env_overrides", None)
        data["reward"] = RewardConfig.from_dict(data.pop("reward", {}))
        data["network"] = NetworkConfig.from_dict(data.pop("network", {}))
        data["rollout"] = RolloutConfig.from_dict(data.pop("rollout", {}))
        data["ppo"] = PPOConfig.from_dict(data.pop("ppo", {}))
        data["logging"] = LoggingConfig.from_dict(data.pop("logging", {}))
        return cls(**data)

    @property
    def input_channels(self) -> int:
        return int(self.observation_shape[0])


def load_stage_config(
    stage: str,
    *,
    scenario_name: str | None = None,
    raw_config: dict | None = None,
) -> PPOTraditionalConfig:
    loaded = RAW_CONFIG if raw_config is None else raw_config
    scenario_name_to_use = scenario_name or get_active_scenario_name(raw_config=loaded)
    _, _, features, observation_shape = _get_observation_metadata(
        stage=stage,
        scenario_name=scenario_name_to_use,
        raw_config=loaded,
    )
    n_actions = _get_action_count(
        stage=stage,
        scenario_name=scenario_name_to_use,
        raw_config=loaded,
    )
    preset = get_stage_preset_config(
        stage=stage,
        scenario_name=scenario_name_to_use,
        raw_config=loaded,
    )
    return PPOTraditionalConfig.from_dict(
        preset,
        default_observation_shape=observation_shape,
        default_observation_features=features,
        default_n_actions=n_actions,
    )


def reload_settings() -> tuple[PPOTraditionalConfig, PPOTraditionalConfig]:
    global CONFIG_PATH, RAW_CONFIG, ACTIVE_SCENARIO
    global DEFAULT_GRID_EXTENT, DEFAULT_GRID_STEP, DEFAULT_FEATURES, DEFAULT_OBSERVATION_SHAPE
    global DEFAULT_N_ACTIONS
    global TRAIN_CONFIG, EVALUATION_CONFIG

    CONFIG_PATH = get_config_path()
    RAW_CONFIG = load_runtime_config(CONFIG_PATH, reload=True)
    ACTIVE_SCENARIO = get_active_scenario_name(raw_config=RAW_CONFIG)
    (
        DEFAULT_GRID_EXTENT,
        DEFAULT_GRID_STEP,
        DEFAULT_FEATURES,
        DEFAULT_OBSERVATION_SHAPE,
    ) = _get_observation_metadata(
        stage="train",
        scenario_name=ACTIVE_SCENARIO,
        raw_config=RAW_CONFIG,
    )
    DEFAULT_N_ACTIONS = _get_action_count(
        stage="train",
        scenario_name=ACTIVE_SCENARIO,
        raw_config=RAW_CONFIG,
    )
    TRAIN_CONFIG = load_stage_config("train", raw_config=RAW_CONFIG)
    EVALUATION_CONFIG = load_stage_config("evaluation", raw_config=RAW_CONFIG)
    return TRAIN_CONFIG, EVALUATION_CONFIG


TRAIN_CONFIG = load_stage_config("train", raw_config=RAW_CONFIG)
EVALUATION_CONFIG = load_stage_config("evaluation", raw_config=RAW_CONFIG)

__all__ = [
    "ACTIVE_SCENARIO",
    "CONFIG_PATH",
    "EVALUATION_CONFIG",
    "LoggingConfig",
    "NetworkConfig",
    "PPOConfig",
    "PPOTraditionalConfig",
    "RAW_CONFIG",
    "RewardConfig",
    "RolloutConfig",
    "TRAIN_CONFIG",
    "compute_grid_shape",
    "load_stage_config",
    "reload_settings",
]

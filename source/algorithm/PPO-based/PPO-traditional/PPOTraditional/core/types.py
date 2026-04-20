from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    fitness: float
    collided: bool
    success: bool
    distance_travelled: float
    episode_length: int
    mean_speed_mps: float
    mean_speed_kph: float
    mean_normalized_speed: float
    mean_low_speed_ratio: float
    mean_step_reward: float
    mean_raw_env_reward: float
    offroad_rate: float
    terminated: bool
    truncated: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RolloutBatch:
    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_metrics: tuple[EpisodeMetrics, ...]
    mean_speed_mps: float
    mean_speed_kph: float
    mean_normalized_speed: float
    mean_low_speed_ratio: float
    mean_step_reward: float
    mean_raw_env_reward: float
    offroad_rate: float
    finished_episode_count: int
    action_counts: dict[str, int] = field(default_factory=dict)
    action_fractions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["episode_metrics"] = [metrics.to_dict() for metrics in self.episode_metrics]
        return payload


@dataclass(frozen=True)
class UpdateSummary:
    update: int
    total_timesteps: int
    best_fitness: float
    mean_fitness: float
    ema_fitness: float
    best_ema_fitness: float
    learning_rate: float
    collision_rate: float
    success_rate: float
    mean_distance: float
    mean_episode_length: float
    mean_speed_mps: float
    mean_speed_kph: float
    mean_normalized_speed: float
    mean_low_speed_ratio: float
    mean_step_reward: float
    mean_raw_env_reward: float
    offroad_rate: float
    finished_episode_count: int
    sample_count: int
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    action_counts: dict[str, int] = field(default_factory=dict)
    action_fractions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["EpisodeMetrics", "RolloutBatch", "UpdateSummary"]

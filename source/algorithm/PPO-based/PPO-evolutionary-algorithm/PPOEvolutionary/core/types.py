from __future__ import annotations

from dataclasses import asdict, dataclass

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
    mean_step_reward: float
    mean_raw_env_reward: float
    offroad_rate: float
    terminated: bool
    truncated: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryBatch:
    obs: np.ndarray
    actions: np.ndarray
    behavior_log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    last_obs: np.ndarray
    last_value: float
    episode_metrics: EpisodeMetrics
    policy_index: int
    seed: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["episode_metrics"] = self.episode_metrics.to_dict()
        return payload


@dataclass(frozen=True)
class GenerationSummary:
    generation: int
    best_fitness: float
    mean_fitness: float
    fitness_std: float
    collision_rate: float
    success_rate: float
    mean_distance: float
    mean_episode_length: float
    mean_speed_mps: float
    mean_speed_kph: float
    mean_normalized_speed: float
    mean_step_reward: float
    mean_raw_env_reward: float
    offroad_rate: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    distill_loss: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["EpisodeMetrics", "GenerationSummary", "TrajectoryBatch"]

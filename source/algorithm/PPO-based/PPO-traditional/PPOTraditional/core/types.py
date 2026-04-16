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
    collision_rate: float
    success_rate: float
    mean_distance: float
    mean_episode_length: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["EpisodeMetrics", "RolloutBatch", "UpdateSummary"]

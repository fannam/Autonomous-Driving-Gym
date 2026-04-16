from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from ..core.types import TrajectoryBatch


@dataclass(frozen=True)
class PolicyAggregate:
    policy_index: int
    fitness: float
    distance_travelled: float
    collision_rate: float
    success_rate: float
    mean_episode_length: float


@dataclass(frozen=True)
class EvolutionStepResult:
    population: list[dict[str, torch.Tensor]]
    elite_indices: tuple[int, ...]
    ranked_policy_indices: tuple[int, ...]
    injected_index: int


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def _is_mutable_parameter(name: str, tensor: torch.Tensor) -> bool:
    if not tensor.is_floating_point():
        return False
    if name.endswith("running_mean") or name.endswith("running_var"):
        return False
    if name.endswith("num_batches_tracked"):
        return False
    return True


def mutate_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    mutation_std: float,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    mutated = clone_state_dict(state_dict)
    if float(mutation_std) == 0.0:
        return mutated

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))

    for name, value in mutated.items():
        if not _is_mutable_parameter(name, value):
            continue
        noise = torch.randn(
            value.shape,
            generator=generator,
            dtype=value.dtype,
            device=value.device,
        )
        value.add_(noise * float(mutation_std))
    return mutated


def initialize_population(
    base_state_dict: dict[str, torch.Tensor],
    *,
    population_size: int,
    mutation_std: float,
    seed: int = 0,
) -> list[dict[str, torch.Tensor]]:
    if int(population_size) <= 0:
        raise ValueError("population_size must be positive.")
    population = [clone_state_dict(base_state_dict)]
    for index in range(1, int(population_size)):
        population.append(
            mutate_state_dict(
                base_state_dict,
                mutation_std=float(mutation_std),
                seed=int(seed) + index,
            )
        )
    return population


def aggregate_policy_metrics(
    trajectories: Iterable[TrajectoryBatch],
    population_size: int,
) -> list[PolicyAggregate]:
    grouped: dict[int, list[TrajectoryBatch]] = {
        int(policy_index): []
        for policy_index in range(int(population_size))
    }
    for trajectory in trajectories:
        grouped.setdefault(int(trajectory.policy_index), []).append(trajectory)

    aggregates: list[PolicyAggregate] = []
    for policy_index in range(int(population_size)):
        policy_trajectories = grouped.get(policy_index, [])
        if not policy_trajectories:
            aggregates.append(
                PolicyAggregate(
                    policy_index=policy_index,
                    fitness=float("-inf"),
                    distance_travelled=float("-inf"),
                    collision_rate=1.0,
                    success_rate=0.0,
                    mean_episode_length=0.0,
                )
            )
            continue

        metrics = [trajectory.episode_metrics for trajectory in policy_trajectories]
        aggregates.append(
            PolicyAggregate(
                policy_index=policy_index,
                fitness=float(np.mean([metric.fitness for metric in metrics])),
                distance_travelled=float(
                    np.mean([metric.distance_travelled for metric in metrics])
                ),
                collision_rate=float(np.mean([float(metric.collided) for metric in metrics])),
                success_rate=float(np.mean([float(metric.success) for metric in metrics])),
                mean_episode_length=float(
                    np.mean([float(metric.episode_length) for metric in metrics])
                ),
            )
        )
    return aggregates


def rank_policy_aggregates(
    aggregates: Iterable[PolicyAggregate],
) -> list[PolicyAggregate]:
    return sorted(
        list(aggregates),
        key=lambda aggregate: (aggregate.fitness, aggregate.distance_travelled),
        reverse=True,
    )


def evolve_population(
    population: list[dict[str, torch.Tensor]],
    ranked_aggregates: list[PolicyAggregate],
    *,
    elite_fraction: float,
    mutation_std: float,
    ppo_state_dict: dict[str, torch.Tensor],
    seed: int = 0,
) -> EvolutionStepResult:
    if not population:
        raise ValueError("Population must not be empty.")
    if len(population) != len(ranked_aggregates):
        raise ValueError("Population and ranked_aggregates must have the same length.")

    population_size = len(population)
    elite_count = max(1, int(math.ceil(population_size * float(elite_fraction))))
    elite_indices = tuple(
        int(aggregate.policy_index)
        for aggregate in ranked_aggregates[:elite_count]
    )
    ranked_policy_indices = tuple(
        int(aggregate.policy_index)
        for aggregate in ranked_aggregates
    )

    new_population = [
        clone_state_dict(population[index])
        for index in elite_indices
    ]

    while len(new_population) < population_size:
        parent_rank = (len(new_population) - elite_count) % elite_count
        parent_index = elite_indices[parent_rank]
        child_seed = int(seed) + len(new_population)
        new_population.append(
            mutate_state_dict(
                population[parent_index],
                mutation_std=float(mutation_std),
                seed=child_seed,
            )
        )

    injected_index = population_size - 1
    new_population[injected_index] = clone_state_dict(ppo_state_dict)
    return EvolutionStepResult(
        population=new_population,
        elite_indices=elite_indices,
        ranked_policy_indices=ranked_policy_indices,
        injected_index=injected_index,
    )


__all__ = [
    "EvolutionStepResult",
    "PolicyAggregate",
    "aggregate_policy_metrics",
    "clone_state_dict",
    "evolve_population",
    "initialize_population",
    "mutate_state_dict",
    "rank_policy_aggregates",
]

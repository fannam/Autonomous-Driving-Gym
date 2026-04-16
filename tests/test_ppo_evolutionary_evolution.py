from __future__ import annotations

import torch

from PPOEvolutionary.core.settings import TRAIN_CONFIG
from PPOEvolutionary.network.actor_critic import build_actor_critic
from PPOEvolutionary.training.evolution import (
    PolicyAggregate,
    evolve_population,
    mutate_state_dict,
)


def test_mutation_preserves_state_shapes_and_dtypes() -> None:
    torch.manual_seed(0)
    state_dict = build_actor_critic(TRAIN_CONFIG).state_dict()
    mutated = mutate_state_dict(state_dict, mutation_std=0.02, seed=7)

    changed_parameter = False
    for name, value in state_dict.items():
        assert mutated[name].shape == value.shape
        assert mutated[name].dtype == value.dtype
        if (
            value.is_floating_point()
            and not name.endswith("running_mean")
            and not name.endswith("running_var")
            and not name.endswith("num_batches_tracked")
            and not torch.equal(mutated[name], value)
        ):
            changed_parameter = True

    assert changed_parameter


def test_evolution_keeps_elites_and_injects_ppo_policy() -> None:
    population = [
        {"weight": torch.tensor([0.0], dtype=torch.float32)},
        {"weight": torch.tensor([1.0], dtype=torch.float32)},
        {"weight": torch.tensor([2.0], dtype=torch.float32)},
        {"weight": torch.tensor([3.0], dtype=torch.float32)},
    ]
    ranked = [
        PolicyAggregate(2, 10.0, 20.0, 0.0, 1.0, 5.0),
        PolicyAggregate(1, 8.0, 15.0, 0.0, 1.0, 5.0),
        PolicyAggregate(3, 6.0, 12.0, 0.0, 1.0, 5.0),
        PolicyAggregate(0, 4.0, 10.0, 1.0, 0.0, 5.0),
    ]
    ppo_state = {"weight": torch.tensor([99.0], dtype=torch.float32)}

    result = evolve_population(
        population,
        ranked,
        elite_fraction=0.25,
        mutation_std=0.0,
        ppo_state_dict=ppo_state,
        seed=13,
    )

    assert result.elite_indices == (2,)
    assert torch.equal(result.population[0]["weight"], population[2]["weight"])
    assert torch.equal(
        result.population[result.injected_index]["weight"],
        ppo_state["weight"],
    )

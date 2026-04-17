from __future__ import annotations

import torch

from PPOEvolutionary.core.settings import TRAIN_CONFIG
from PPOEvolutionary.network.actor_critic import build_actor_critic
from PPOEvolutionary.training.evolution import (
    PolicyAggregate,
    evolve_population,
    mutate_state_dict,
    sync_shared_parameters,
)


def test_mutation_preserves_state_shapes_and_dtypes_and_only_changes_heads() -> None:
    torch.manual_seed(0)
    state_dict = build_actor_critic(TRAIN_CONFIG).state_dict()
    mutated = mutate_state_dict(state_dict, mutation_std=0.02, seed=7)

    changed_head_parameter = False
    for name, value in state_dict.items():
        assert mutated[name].shape == value.shape
        assert mutated[name].dtype == value.dtype
        if name.startswith(("policy_head.", "value_head.")):
            if not torch.equal(mutated[name], value):
                changed_head_parameter = True
        else:
            assert torch.equal(mutated[name], value)

    assert changed_head_parameter


def test_sync_shared_parameters_overwrites_extractor_only() -> None:
    state_dict = {
        "stem.weight": torch.tensor([1.0], dtype=torch.float32),
        "projection.weight": torch.tensor([2.0], dtype=torch.float32),
        "policy_head.weight": torch.tensor([3.0], dtype=torch.float32),
        "value_head.bias": torch.tensor([4.0], dtype=torch.float32),
    }
    shared_state_dict = {
        "stem.weight": torch.tensor([10.0], dtype=torch.float32),
        "projection.weight": torch.tensor([20.0], dtype=torch.float32),
        "policy_head.weight": torch.tensor([30.0], dtype=torch.float32),
        "value_head.bias": torch.tensor([40.0], dtype=torch.float32),
    }

    synced = sync_shared_parameters(
        state_dict,
        shared_state_dict=shared_state_dict,
    )

    assert torch.equal(synced["stem.weight"], shared_state_dict["stem.weight"])
    assert torch.equal(synced["projection.weight"], shared_state_dict["projection.weight"])
    assert torch.equal(synced["policy_head.weight"], state_dict["policy_head.weight"])
    assert torch.equal(synced["value_head.bias"], state_dict["value_head.bias"])


def test_evolution_keeps_elite_heads_syncs_extractor_and_injects_ppo_policy() -> None:
    population = [
        {
            "stem.weight": torch.tensor([0.0], dtype=torch.float32),
            "policy_head.weight": torch.tensor([10.0], dtype=torch.float32),
            "value_head.weight": torch.tensor([20.0], dtype=torch.float32),
        },
        {
            "stem.weight": torch.tensor([1.0], dtype=torch.float32),
            "policy_head.weight": torch.tensor([11.0], dtype=torch.float32),
            "value_head.weight": torch.tensor([21.0], dtype=torch.float32),
        },
        {
            "stem.weight": torch.tensor([2.0], dtype=torch.float32),
            "policy_head.weight": torch.tensor([12.0], dtype=torch.float32),
            "value_head.weight": torch.tensor([22.0], dtype=torch.float32),
        },
        {
            "stem.weight": torch.tensor([3.0], dtype=torch.float32),
            "policy_head.weight": torch.tensor([13.0], dtype=torch.float32),
            "value_head.weight": torch.tensor([23.0], dtype=torch.float32),
        },
    ]
    ranked = [
        PolicyAggregate(2, 10.0, 20.0, 0.0, 1.0, 5.0),
        PolicyAggregate(1, 8.0, 15.0, 0.0, 1.0, 5.0),
        PolicyAggregate(3, 6.0, 12.0, 0.0, 1.0, 5.0),
        PolicyAggregate(0, 4.0, 10.0, 1.0, 0.0, 5.0),
    ]
    ppo_state = {
        "stem.weight": torch.tensor([99.0], dtype=torch.float32),
        "policy_head.weight": torch.tensor([109.0], dtype=torch.float32),
        "value_head.weight": torch.tensor([209.0], dtype=torch.float32),
    }

    result = evolve_population(
        population,
        ranked,
        elite_fraction=0.25,
        mutation_std=0.0,
        ppo_state_dict=ppo_state,
        seed=13,
    )

    assert result.elite_indices == (2,)
    assert torch.equal(result.population[0]["stem.weight"], ppo_state["stem.weight"])
    assert torch.equal(
        result.population[0]["policy_head.weight"],
        population[2]["policy_head.weight"],
    )
    assert torch.equal(
        result.population[0]["value_head.weight"],
        population[2]["value_head.weight"],
    )
    assert torch.equal(
        result.population[result.injected_index]["stem.weight"],
        ppo_state["stem.weight"],
    )
    assert torch.equal(
        result.population[result.injected_index]["policy_head.weight"],
        ppo_state["policy_head.weight"],
    )
    assert torch.equal(
        result.population[result.injected_index]["value_head.weight"],
        ppo_state["value_head.weight"],
    )

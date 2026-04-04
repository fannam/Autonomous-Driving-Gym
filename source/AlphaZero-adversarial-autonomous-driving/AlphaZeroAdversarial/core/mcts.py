from __future__ import annotations

import numpy as np

from autonomous_driving_shared.alphazero_adversarial.core.mcts import (
    BaseSimultaneousMCTS,
    SearchStats,
    SimultaneousMCTSNode,
    _get_device,
    _argmax_action,
    _prune_actions_by_relative_threshold,
    _select_top_actions,
)
from autonomous_driving_shared.alphazero_adversarial.core.game import (
    apply_terminal_value_overrides,
)

from .policy import outer_product_policy


class SimultaneousMCTS(BaseSimultaneousMCTS):
    def __init__(
        self,
        *,
        root: SimultaneousMCTSNode,
        network,
        tensor_builder,
        device=None,
        c_puct: float = 2.5,
        n_simulations: int = 24,
        root_dirichlet_alpha: float = 0.3,
        root_exploration_fraction: float = 0.25,
        max_expand_actions_per_agent: int | None = None,
        n_action_axis_0: int = 5,
        n_action_axis_1: int = 5,
        relative_pruning_gamma: float | None = None,
        flip_npc_steering: bool = True,
    ):
        super().__init__(
            root=root,
            network=network,
            tensor_builder=tensor_builder,
            device=device,
            c_puct=c_puct,
            n_simulations=n_simulations,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_exploration_fraction=root_exploration_fraction,
            max_expand_actions_per_agent=max_expand_actions_per_agent,
            relative_pruning_gamma=relative_pruning_gamma,
        )
        self.n_action_axis_0 = int(n_action_axis_0)
        self.n_action_axis_1 = int(n_action_axis_1)
        self.flip_npc_steering = bool(flip_npc_steering)

    @staticmethod
    def outer_product(
        accelerate_policy,
        steering_policy,
        *,
        n_action_axis_0: int,
        n_action_axis_1: int,
        flip_steering: bool = False,
    ) -> dict[int, float]:
        flat_policy = outer_product_policy(
            accelerate_policy,
            steering_policy,
            n_action_axis_0=n_action_axis_0,
            n_action_axis_1=n_action_axis_1,
            flip_steering=flip_steering,
        )
        return {
            int(action): float(probability)
            for action, probability in enumerate(flat_policy.tolist())
        }

    def _predict(
        self,
        node: SimultaneousMCTSNode,
    ) -> tuple[dict[int, float], dict[int, float], float, float]:
        accelerate_batch, steering_batch, value_batch = self._forward_network(node)

        ego_policy = self.outer_product(
            accelerate_batch[0].tolist(),
            steering_batch[0].tolist(),
            n_action_axis_0=self.n_action_axis_0,
            n_action_axis_1=self.n_action_axis_1,
            flip_steering=False,
        )
        npc_policy = self.outer_product(
            accelerate_batch[1].tolist(),
            steering_batch[1].tolist(),
            n_action_axis_0=self.n_action_axis_0,
            n_action_axis_1=self.n_action_axis_1,
            flip_steering=self.flip_npc_steering,
        )

        ego_value, npc_value = self._clip_value_batch(value_batch)
        ego_value, npc_value = apply_terminal_value_overrides(
            getattr(node, "env", None),
            ego_value,
            npc_value,
        )
        return ego_policy, npc_policy, ego_value, npc_value


__all__ = [
    "SearchStats",
    "SimultaneousMCTS",
    "SimultaneousMCTSNode",
    "_get_device",
    "_argmax_action",
    "_prune_actions_by_relative_threshold",
    "_select_top_actions",
]

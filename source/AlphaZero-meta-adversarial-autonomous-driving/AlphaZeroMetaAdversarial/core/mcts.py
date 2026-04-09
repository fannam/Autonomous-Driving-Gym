from __future__ import annotations

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

from .action_score_heuristics import (
    ActionScoreHeuristic,
    build_default_action_score_heuristic,
)


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
        relative_pruning_gamma: float | None = None,
        discount_gamma: float = 1.0,
        npc_closing_ucb_bonus: float = 0.0,
        action_score_heuristic: ActionScoreHeuristic | None = None,
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
            discount_gamma=discount_gamma,
        )
        self.action_score_heuristic = (
            action_score_heuristic
            if action_score_heuristic is not None
            else build_default_action_score_heuristic(
                npc_closing_ucb_bonus=float(npc_closing_ucb_bonus),
            )
        )

    def _predict(
        self,
        node: SimultaneousMCTSNode,
    ) -> tuple[dict[int, float], dict[int, float], float, float]:
        policy_batch, value_batch = self._forward_network(node)

        ego_policy = {action: prob for action, prob in enumerate(policy_batch[0].tolist())}
        npc_policy = {action: prob for action, prob in enumerate(policy_batch[1].tolist())}

        ego_value, npc_value = self._clip_value_batch(value_batch)
        ego_value, npc_value = apply_terminal_value_overrides(
            getattr(node, "env", None),
            ego_value,
            npc_value,
        )
        return ego_policy, npc_policy, ego_value, npc_value

    def _action_score_bonuses(
        self,
        node: SimultaneousMCTSNode,
    ) -> tuple[dict[int, float], dict[int, float]]:
        if self.action_score_heuristic is None:
            return {}, {}
        return self.action_score_heuristic.action_score_bonuses(
            node=node,
            tensor_builder=self._builder,
        )


__all__ = [
    "SearchStats",
    "SimultaneousMCTS",
    "SimultaneousMCTSNode",
    "_get_device",
    "_argmax_action",
    "_prune_actions_by_relative_threshold",
    "_select_top_actions",
]

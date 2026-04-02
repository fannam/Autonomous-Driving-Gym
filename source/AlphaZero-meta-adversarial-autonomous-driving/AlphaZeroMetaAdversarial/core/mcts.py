from __future__ import annotations

from autonomous_driving_shared.alphazero_adversarial.core.mcts import (
    BaseSimultaneousMCTS,
    SearchStats,
    SimultaneousMCTSNode,
    _argmax_action,
    _prune_actions_by_relative_threshold,
    _resolve_device,
    _select_top_actions,
)


class SimultaneousMCTS(BaseSimultaneousMCTS):
    def _predict(
        self,
        node: SimultaneousMCTSNode,
    ) -> tuple[dict[int, float], dict[int, float], float, float]:
        policy_batch, value_batch = self._forward_network(node)

        ego_policy = {action: prob for action, prob in enumerate(policy_batch[0].tolist())}
        npc_policy = {action: prob for action, prob in enumerate(policy_batch[1].tolist())}

        ego_value, npc_value = self._clip_value_batch(value_batch)
        return ego_policy, npc_policy, ego_value, npc_value


__all__ = [
    "SearchStats",
    "SimultaneousMCTS",
    "SimultaneousMCTSNode",
    "_argmax_action",
    "_prune_actions_by_relative_threshold",
    "_resolve_device",
    "_select_top_actions",
]

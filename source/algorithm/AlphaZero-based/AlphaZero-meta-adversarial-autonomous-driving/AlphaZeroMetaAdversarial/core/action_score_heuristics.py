from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autonomous_driving_shared.alphazero_adversarial.core.game import (
    get_controlled_vehicles,
)
from autonomous_driving_shared.alphazero_adversarial.core.mcts import (
    SimultaneousMCTSNode,
)
from autonomous_driving_shared.alphazero_adversarial.core.perspective_stack import (
    PerspectiveTensorBuilder,
)


class ActionScoreHeuristic:
    def action_score_bonuses(
        self,
        *,
        node: SimultaneousMCTSNode,
        tensor_builder: PerspectiveTensorBuilder,
    ) -> tuple[dict[int, float], dict[int, float]]:
        del node
        del tensor_builder
        return {}, {}


@dataclass(frozen=True)
class CompositeActionScoreHeuristic(ActionScoreHeuristic):
    heuristics: tuple[ActionScoreHeuristic, ...] = ()

    def action_score_bonuses(
        self,
        *,
        node: SimultaneousMCTSNode,
        tensor_builder: PerspectiveTensorBuilder,
    ) -> tuple[dict[int, float], dict[int, float]]:
        ego_bonus_by_action: dict[int, float] = {}
        npc_bonus_by_action: dict[int, float] = {}

        for heuristic in self.heuristics:
            ego_bonuses, npc_bonuses = heuristic.action_score_bonuses(
                node=node,
                tensor_builder=tensor_builder,
            )
            _merge_action_bonus_maps(ego_bonus_by_action, ego_bonuses)
            _merge_action_bonus_maps(npc_bonus_by_action, npc_bonuses)
        return ego_bonus_by_action, npc_bonus_by_action


@dataclass(frozen=True)
class NPCClosingActionScoreHeuristic(ActionScoreHeuristic):
    bonus_scale: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.bonus_scale)) or float(self.bonus_scale) < 0.0:
            raise ValueError("bonus_scale must be a non-negative finite float.")

    def action_score_bonuses(
        self,
        *,
        node: SimultaneousMCTSNode,
        tensor_builder: PerspectiveTensorBuilder,
    ) -> tuple[dict[int, float], dict[int, float]]:
        if self.bonus_scale <= 0.0:
            return {}, {}

        npc_bonus_by_action = {
            int(action): self._npc_bonus_for_action(
                node=node,
                action=int(action),
                tensor_builder=tensor_builder,
            )
            for action in node.npc_available_actions
        }
        return {}, npc_bonus_by_action

    @staticmethod
    def _get_action_label(
        node: SimultaneousMCTSNode,
        *,
        agent_index: int,
        action: int,
    ) -> str | None:
        env = getattr(node, "env", None)
        if env is None:
            return None
        env_unwrapped = getattr(env, "unwrapped", env)
        action_type = getattr(env_unwrapped, "action_type", None)
        agents_action_types = getattr(action_type, "agents_action_types", None)
        if agents_action_types is None or len(agents_action_types) <= agent_index:
            return None
        actions = getattr(agents_action_types[agent_index], "actions", None)
        if not isinstance(actions, dict):
            return None
        label = actions.get(int(action))
        return None if label is None else str(label)

    def _npc_bonus_for_action(
        self,
        *,
        node: SimultaneousMCTSNode,
        action: int,
        tensor_builder: PerspectiveTensorBuilder,
    ) -> float:
        env = getattr(node, "env", None)
        if env is None:
            return 0.0

        action_label = self._get_action_label(node, agent_index=1, action=action)
        if action_label not in {"LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER"}:
            return 0.0

        try:
            ego_vehicle, npc_vehicle = get_controlled_vehicles(env)
            intercept_position, _ = tensor_builder._get_intercept_target(
                self_vehicle=npc_vehicle,
                target_vehicle=ego_vehicle,
            )
        except Exception:
            return 0.0

        ego_lane_index = getattr(ego_vehicle, "target_lane_index", None) or getattr(
            ego_vehicle,
            "lane_index",
            None,
        )
        npc_lane_index = getattr(npc_vehicle, "target_lane_index", None) or getattr(
            npc_vehicle,
            "lane_index",
            None,
        )

        if action_label in {"LANE_LEFT", "LANE_RIGHT"}:
            if (
                ego_lane_index is None
                or npc_lane_index is None
                or len(ego_lane_index) < 3
                or len(npc_lane_index) < 3
                or ego_lane_index[:2] != npc_lane_index[:2]
            ):
                return 0.0
            lane_delta = int(ego_lane_index[2]) - int(npc_lane_index[2])
            if lane_delta == 0:
                return 0.0
            if action_label == "LANE_RIGHT" and lane_delta > 0:
                return self.bonus_scale * float(min(1.0, abs(lane_delta)))
            if action_label == "LANE_LEFT" and lane_delta < 0:
                return self.bonus_scale * float(min(1.0, abs(lane_delta)))
            return 0.0

        npc_position = np.asarray(
            getattr(npc_vehicle, "position", np.zeros(2, dtype=np.float32)),
            dtype=np.float32,
        )
        heading = float(getattr(npc_vehicle, "heading", 0.0))
        forward_axis = np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float32)
        intercept_gap = float(
            np.dot(
                np.asarray(intercept_position, dtype=np.float32) - npc_position,
                forward_axis,
            )
        )
        if np.isclose(intercept_gap, 0.0):
            return 0.0

        longitudinal_scale = max(
            float(getattr(tensor_builder.config, "route_lookahead_base", 20.0)),
            1e-6,
        )
        speed_urgency = float(np.tanh(abs(intercept_gap) / longitudinal_scale))
        if action_label == "FASTER" and intercept_gap > 0.0:
            return self.bonus_scale * speed_urgency
        if action_label == "SLOWER" and intercept_gap < 0.0:
            return self.bonus_scale * speed_urgency
        return 0.0


def build_default_action_score_heuristic(
    *,
    npc_closing_ucb_bonus: float = 0.0,
) -> ActionScoreHeuristic | None:
    if (
        not np.isfinite(float(npc_closing_ucb_bonus))
        or float(npc_closing_ucb_bonus) < 0.0
    ):
        raise ValueError("npc_closing_ucb_bonus must be a non-negative finite float.")

    heuristics: list[ActionScoreHeuristic] = []
    if float(npc_closing_ucb_bonus) > 0.0:
        heuristics.append(
            NPCClosingActionScoreHeuristic(
                bonus_scale=float(npc_closing_ucb_bonus),
            )
        )

    if not heuristics:
        return None
    if len(heuristics) == 1:
        return heuristics[0]
    return CompositeActionScoreHeuristic(tuple(heuristics))


def _merge_action_bonus_maps(
    target: dict[int, float],
    source: dict[int, float],
) -> None:
    for action, bonus in source.items():
        target[int(action)] = float(target.get(int(action), 0.0)) + float(bonus)


__all__ = [
    "ActionScoreHeuristic",
    "CompositeActionScoreHeuristic",
    "NPCClosingActionScoreHeuristic",
    "build_default_action_score_heuristic",
]

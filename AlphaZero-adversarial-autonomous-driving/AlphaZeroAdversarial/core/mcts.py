from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass

import numpy as np
import torch

from .game import (
    TerminalOutcome,
    classify_terminal_state,
    get_available_actions,
)
from .perspective_stack import (
    PerspectiveHistory,
    PerspectiveTensorBuilder,
    advance_history,
    seed_history_from_env,
)
from .policy import normalize_policy
from .settings import PerspectiveTensorConfig, ZeroSumConfig


def _resolve_device(device=None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but CUDA is not available.")
    return resolved


def _select_top_actions(
    available_actions: tuple[int, ...],
    policy: dict[int, float],
    limit: int | None,
) -> tuple[int, ...]:
    if limit is None or len(available_actions) <= limit:
        return available_actions
    ranked = sorted(
        available_actions,
        key=lambda action: float(policy.get(action, 0.0)),
        reverse=True,
    )
    return tuple(ranked[:limit])


def _argmax_action(scores: dict[int, float]) -> int:
    if not scores:
        raise ValueError("Expected at least one score entry.")
    best_score = max(scores.values())
    best_actions = [action for action, score in scores.items() if score == best_score]
    return int(np.random.choice(best_actions))


@dataclass
class SearchStats:
    rollouts: int = 0
    inference_calls: int = 0
    terminal_rollouts: int = 0
    rollout_time_s: float = 0.0
    inference_time_s: float = 0.0


class SimultaneousMCTSNode:
    def __init__(
        self,
        env,
        parent,
        parent_joint_action: tuple[int, int] | None,
        *,
        tensor_config: PerspectiveTensorConfig,
        zero_sum_config: ZeroSumConfig,
        agent_policy_modes: tuple[str | None, str | None] = (None, None),
        history: PerspectiveHistory | None = None,
        copy_env: bool = True,
    ):
        self.env = copy.deepcopy(env) if (copy_env and env is not None) else env
        self.parent = parent
        self.parent_joint_action = parent_joint_action
        self.tensor_config = tensor_config
        self.zero_sum_config = zero_sum_config
        self.agent_policy_modes = agent_policy_modes
        self.children: dict[tuple[int, int], "SimultaneousMCTSNode"] = {}
        self.visit_count = 0

        self.ego_available_actions: tuple[int, ...] = tuple()
        self.npc_available_actions: tuple[int, ...] = tuple()
        self.ego_action_priors: dict[int, float] = {}
        self.npc_action_priors: dict[int, float] = {}
        self.ego_action_visits: dict[int, int] = {}
        self.npc_action_visits: dict[int, int] = {}
        self.ego_action_value_sums: dict[int, float] = {}
        self.npc_action_value_sums: dict[int, float] = {}
        self.terminal_outcome = TerminalOutcome(False, 0.0, 0.0, "uninitialized")

        self.history = history
        self.cached_perspective_batch: np.ndarray | None = None

        if self.env is not None:
            self._refresh_from_env()
            if self.history is None:
                if self.parent is None:
                    self.history = seed_history_from_env(self.env, self.tensor_config.history_length)
                else:
                    if self.parent.history is None:
                        raise RuntimeError("Parent history must exist before child history can be derived.")
                    self.history = advance_history(
                        self.parent.history,
                        self.env,
                        self.tensor_config.history_length,
                    )

    def _refresh_from_env(self) -> None:
        self.terminal_outcome = classify_terminal_state(self.env, self.zero_sum_config)
        if self.terminal_outcome.terminal:
            self.ego_available_actions = tuple()
            self.npc_available_actions = tuple()
            return
        self.ego_available_actions, self.npc_available_actions = get_available_actions(
            self.env,
            policy_modes=self.agent_policy_modes,
        )

    def ensure_instantiated(self) -> None:
        if self.env is not None:
            return
        if self.parent is None or self.parent_joint_action is None:
            raise RuntimeError("Cannot instantiate a root node without an env snapshot.")

        self.parent.ensure_instantiated()
        next_env = copy.deepcopy(self.parent.env)
        env_unwrapped = getattr(next_env, "unwrapped", next_env)
        step_for_mcts = getattr(env_unwrapped, "step_for_mcts", None)
        if callable(step_for_mcts):
            step_for_mcts(self.parent_joint_action)
        else:
            next_env.step(self.parent_joint_action)
        self.env = next_env
        self._refresh_from_env()
        if self.parent.history is None:
            raise RuntimeError("Parent history must be initialized before child instantiation.")
        self.history = advance_history(
            self.parent.history,
            self.env,
            self.tensor_config.history_length,
        )

    def ensure_perspective_batch(self, builder: PerspectiveTensorBuilder) -> None:
        self.ensure_instantiated()
        if self.cached_perspective_batch is not None:
            return
        if self.history is None:
            raise RuntimeError("Node history must be initialized before building tensors.")
        self.cached_perspective_batch = builder.build_batch(self.env, self.history)

    def is_leaf(self) -> bool:
        return not self.children

    def _puct_score(
        self,
        *,
        action: int,
        priors: dict[int, float],
        visits: dict[int, int],
        value_sums: dict[int, float],
        c_puct: float,
    ) -> float:
        action_visits = int(visits.get(action, 0))
        q_value = 0.0 if action_visits == 0 else float(value_sums.get(action, 0.0)) / action_visits
        prior = float(priors.get(action, 0.0))
        exploration = c_puct * prior * np.sqrt(max(1, self.visit_count)) / (1 + action_visits)
        return q_value + exploration

    def select_joint_action(self, c_puct: float) -> tuple[int, int]:
        ego_scores = {
            action: self._puct_score(
                action=action,
                priors=self.ego_action_priors,
                visits=self.ego_action_visits,
                value_sums=self.ego_action_value_sums,
                c_puct=c_puct,
            )
            for action in self.ego_available_actions
        }
        npc_scores = {
            action: self._puct_score(
                action=action,
                priors=self.npc_action_priors,
                visits=self.npc_action_visits,
                value_sums=self.npc_action_value_sums,
                c_puct=c_puct,
            )
            for action in self.npc_available_actions
        }
        return _argmax_action(ego_scores), _argmax_action(npc_scores)

    def expand(
        self,
        *,
        ego_policy: dict[int, float],
        npc_policy: dict[int, float],
        max_expand_actions_per_agent: int | None,
    ) -> None:
        selected_ego_actions = _select_top_actions(
            self.ego_available_actions,
            ego_policy,
            max_expand_actions_per_agent,
        )
        selected_npc_actions = _select_top_actions(
            self.npc_available_actions,
            npc_policy,
            max_expand_actions_per_agent,
        )
        self.ego_available_actions = selected_ego_actions
        self.npc_available_actions = selected_npc_actions
        self.ego_action_priors = normalize_policy(ego_policy, self.ego_available_actions)
        self.npc_action_priors = normalize_policy(npc_policy, self.npc_available_actions)

        for action in self.ego_available_actions:
            self.ego_action_visits.setdefault(action, 0)
            self.ego_action_value_sums.setdefault(action, 0.0)
        for action in self.npc_available_actions:
            self.npc_action_visits.setdefault(action, 0)
            self.npc_action_value_sums.setdefault(action, 0.0)

        for joint_action in itertools.product(self.ego_available_actions, self.npc_available_actions):
            if joint_action in self.children:
                continue
            self.children[joint_action] = SimultaneousMCTSNode(
                env=None,
                parent=self,
                parent_joint_action=(int(joint_action[0]), int(joint_action[1])),
                tensor_config=self.tensor_config,
                zero_sum_config=self.zero_sum_config,
                agent_policy_modes=self.agent_policy_modes,
                history=None,
                copy_env=False,
            )

    def backpropagate_recursive(self, ego_value: float, npc_value: float) -> None:
        current = self
        while current is not None:
            current.visit_count += 1
            parent = current.parent
            if parent is not None and current.parent_joint_action is not None:
                ego_action, npc_action = current.parent_joint_action
                parent.ego_action_visits[ego_action] = parent.ego_action_visits.get(ego_action, 0) + 1
                parent.npc_action_visits[npc_action] = parent.npc_action_visits.get(npc_action, 0) + 1
                parent.ego_action_value_sums[ego_action] = (
                    parent.ego_action_value_sums.get(ego_action, 0.0) + float(ego_value)
                )
                parent.npc_action_value_sums[npc_action] = (
                    parent.npc_action_value_sums.get(npc_action, 0.0) + float(npc_value)
                )
            current = parent


class SimultaneousMCTS:
    def __init__(
        self,
        *,
        root: SimultaneousMCTSNode,
        network,
        tensor_builder: PerspectiveTensorBuilder,
        device=None,
        c_puct: float = 2.5,
        n_simulations: int = 24,
        root_dirichlet_alpha: float = 0.3,
        root_exploration_fraction: float = 0.25,
        max_expand_actions_per_agent: int | None = None,
    ):
        self.root = root
        self.c_puct = float(c_puct)
        self.n_simulations = int(n_simulations)
        self.root_dirichlet_alpha = float(root_dirichlet_alpha)
        self.root_exploration_fraction = float(root_exploration_fraction)
        self.max_expand_actions_per_agent = (
            None
            if max_expand_actions_per_agent is None
            else int(max_expand_actions_per_agent)
        )
        self._device = _resolve_device(device)
        self._network = network.to(self._device)
        if self._network.training:
            self._network.eval()
        self._builder = tensor_builder
        self._pending_root_noise = False
        self._stats = SearchStats()

    def reset_timing_stats(self) -> None:
        self._stats = SearchStats()

    def get_timing_stats(self) -> dict[str, float | int]:
        rollouts = int(self._stats.rollouts)
        inference_calls = int(self._stats.inference_calls)
        rollout_time_s = float(self._stats.rollout_time_s)
        inference_time_s = float(self._stats.inference_time_s)
        return {
            "rollouts": rollouts,
            "inference_calls": inference_calls,
            "terminal_rollouts": int(self._stats.terminal_rollouts),
            "rollout_time_s": rollout_time_s,
            "inference_time_s": inference_time_s,
            "avg_rollout_ms": 0.0 if rollouts == 0 else 1000.0 * rollout_time_s / rollouts,
            "avg_inference_ms": (
                0.0 if inference_calls == 0 else 1000.0 * inference_time_s / inference_calls
            ),
            "rollouts_per_sec": 0.0 if rollout_time_s <= 0.0 else rollouts / rollout_time_s,
        }

    def traverse_to_leaf(self) -> tuple[SimultaneousMCTSNode, int]:
        node = self.root
        depth = 0
        while not node.is_leaf():
            joint_action = node.select_joint_action(self.c_puct)
            node = node.children[joint_action]
            depth += 1
        return node, depth

    def _predict(self, node: SimultaneousMCTSNode) -> tuple[dict[int, float], dict[int, float], float, float]:
        node.ensure_perspective_batch(self._builder)
        batch = torch.from_numpy(node.cached_perspective_batch).to(
            device=self._device,
            dtype=torch.float32,
            non_blocking=self._device.type != "cpu",
        )
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        started_at = time.perf_counter()
        with torch.inference_mode():
            policy_batch, value_batch = self._network(batch)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        elapsed = time.perf_counter() - started_at
        self._stats.inference_calls += 1
        self._stats.inference_time_s += elapsed

        ego_policy = {action: prob for action, prob in enumerate(policy_batch[0].tolist())}
        npc_policy = {action: prob for action, prob in enumerate(policy_batch[1].tolist())}

        raw_ego_value = float(value_batch[0].item())
        raw_npc_value = float(value_batch[1].item())
        ego_value = float(np.clip(0.5 * (raw_ego_value - raw_npc_value), -1.0, 1.0))
        npc_value = -ego_value
        return ego_policy, npc_policy, ego_value, npc_value

    def _apply_dirichlet_noise(
        self,
        policy: dict[int, float],
        available_actions: tuple[int, ...],
    ) -> dict[int, float]:
        if not available_actions:
            return policy
        noise = np.random.dirichlet(
            [self.root_dirichlet_alpha] * len(available_actions)
        )
        mixed_policy = dict(policy)
        epsilon = self.root_exploration_fraction
        for action, noise_prob in zip(available_actions, noise):
            prior = float(policy.get(action, 0.0))
            mixed_policy[action] = (1.0 - epsilon) * prior + epsilon * float(noise_prob)
        return normalize_policy(mixed_policy, available_actions)

    def prepare_root(self, add_exploration_noise: bool = False) -> None:
        self._pending_root_noise = bool(add_exploration_noise)
        if not self._pending_root_noise or self.root.is_leaf():
            return

        self.root.ego_action_priors = self._apply_dirichlet_noise(
            self.root.ego_action_priors,
            self.root.ego_available_actions,
        )
        self.root.npc_action_priors = self._apply_dirichlet_noise(
            self.root.npc_action_priors,
            self.root.npc_available_actions,
        )
        self._pending_root_noise = False

    def rollout(self) -> None:
        rollout_started_at = time.perf_counter()
        try:
            leaf, _ = self.traverse_to_leaf()
            leaf.ensure_instantiated()
            if leaf.terminal_outcome.terminal:
                self._stats.terminal_rollouts += 1
                leaf.backpropagate_recursive(
                    leaf.terminal_outcome.ego_value,
                    leaf.terminal_outcome.npc_value,
                )
                return

            ego_policy, npc_policy, ego_value, npc_value = self._predict(leaf)
            ego_policy = normalize_policy(ego_policy, leaf.ego_available_actions)
            npc_policy = normalize_policy(npc_policy, leaf.npc_available_actions)

            if leaf is self.root and self._pending_root_noise:
                ego_policy = self._apply_dirichlet_noise(
                    ego_policy,
                    leaf.ego_available_actions,
                )
                npc_policy = self._apply_dirichlet_noise(
                    npc_policy,
                    leaf.npc_available_actions,
                )
                self._pending_root_noise = False

            leaf.expand(
                ego_policy=ego_policy,
                npc_policy=npc_policy,
                max_expand_actions_per_agent=self.max_expand_actions_per_agent,
            )
            leaf.backpropagate_recursive(ego_value, npc_value)
        finally:
            self._stats.rollouts += 1
            self._stats.rollout_time_s += time.perf_counter() - rollout_started_at

    def action_distribution(self, *, agent_index: int, temperature: float) -> dict[int, float]:
        if agent_index == 0:
            available_actions = self.root.ego_available_actions
            visits = self.root.ego_action_visits
        else:
            available_actions = self.root.npc_available_actions
            visits = self.root.npc_action_visits

        if not available_actions:
            return {}

        visit_counts = np.asarray(
            [float(visits.get(action, 0)) for action in available_actions],
            dtype=np.float32,
        )
        if float(np.sum(visit_counts)) <= 0.0:
            uniform_prob = 1.0 / len(available_actions)
            return {int(action): uniform_prob for action in available_actions}

        if temperature <= 1e-8:
            best_visit = float(np.max(visit_counts))
            best_actions = [
                int(action)
                for action, count in zip(available_actions, visit_counts)
                if float(count) == best_visit
            ]
            chosen_action = int(np.random.choice(best_actions))
            return {int(action): 1.0 if int(action) == chosen_action else 0.0 for action in available_actions}

        scaled_counts = np.power(visit_counts, 1.0 / float(temperature))
        scaled_sum = float(np.sum(scaled_counts))
        if scaled_sum <= 0.0 or not np.isfinite(scaled_sum):
            uniform_prob = 1.0 / len(available_actions)
            return {int(action): uniform_prob for action in available_actions}
        scaled_counts /= scaled_sum
        return {
            int(action): float(prob)
            for action, prob in zip(available_actions, scaled_counts)
        }

    def move_to_new_root(self, joint_action: tuple[int, int]) -> None:
        if joint_action not in self.root.children:
            raise ValueError("Joint action is not available in the current tree.")
        old_root = self.root
        new_root = old_root.children[joint_action]
        new_root.ensure_instantiated()
        new_root.parent = None
        old_root.children.clear()
        self.root = new_root
        self._pending_root_noise = False

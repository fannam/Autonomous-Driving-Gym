from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..core.game import (
    TerminalOutcome,
    classify_terminal_state,
    get_agent_snapshots,
    get_available_actions,
    get_progress_value,
)
from ..core.mcts import SimultaneousMCTSNode
from ..core.perspective_stack import PerspectiveTensorBuilder


def _resolve_training_device(device) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but CUDA is not available.")
    return resolved


@dataclass(frozen=True)
class EpisodeStepSample:
    state: np.ndarray
    target_vector: np.ndarray
    policy_targets: tuple[np.ndarray, ...]
    agent_index: int


class BaseAdversarialAlphaZeroTrainer:
    def __init__(
        self,
        *,
        network,
        config,
        env=None,
        device="auto",
        verbose: bool = True,
        reuse_tree_between_steps: bool = True,
        add_root_dirichlet_noise: bool = True,
    ):
        self.config = config
        self.device = _resolve_training_device(device)
        self.network = network.to(self.device)
        self.env = env
        self.verbose = bool(verbose)
        self.reuse_tree_between_steps = bool(reuse_tree_between_steps)
        self.add_root_dirichlet_noise = bool(add_root_dirichlet_noise)
        self.tensor_builder = PerspectiveTensorBuilder(config.tensor)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.replay_buffer = deque(maxlen=int(config.replay_buffer_size))
        self.last_episode_outcome: TerminalOutcome | None = None
        self.last_episode_summary: dict | None = None

    @staticmethod
    def _kl_policy_loss(policy_logits, policy_targets):
        normalized_targets = policy_targets / policy_targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
        log_policy = F.log_softmax(policy_logits, dim=1)
        return F.kl_div(log_policy, normalized_targets, reduction="batchmean")

    def _require_env(self):
        if self.env is None:
            raise RuntimeError("This trainer operation requires an initialized environment.")
        return self.env

    def _temperature_for_step(self, step_count: int) -> float:
        if self.config.temperature_drop_step is not None and step_count >= self.config.temperature_drop_step:
            return 0.0
        return float(self.config.temperature)

    def _policy_modes_for_episode(self, episode_index: int) -> tuple[str | None, str | None]:
        if episode_index < int(self.config.warmup_episodes):
            return (None, self.config.warmup_opponent_policy)
        return (None, None)

    def _should_collect_npc_samples(self, episode_index: int) -> bool:
        if episode_index < int(self.config.warmup_episodes):
            return bool(self.config.warmup_collect_opponent_samples)
        return True

    def _build_root_node(
        self,
        *,
        policy_modes: tuple[str | None, str | None],
    ) -> SimultaneousMCTSNode:
        return SimultaneousMCTSNode(
            env=self._require_env(),
            parent=None,
            parent_joint_action=None,
            tensor_config=self.config.tensor,
            zero_sum_config=self.config.zero_sum,
            agent_policy_modes=policy_modes,
            history=None,
            copy_env=False,
        )

    def _build_mcts(self, root_node: SimultaneousMCTSNode):
        raise NotImplementedError

    def _run_rollouts(self, mcts) -> int:
        for _ in range(int(self.config.n_simulations)):
            mcts.rollout()
        return int(self.config.n_simulations)

    @staticmethod
    def _sample_action(policy_dict: dict[int, float]) -> int:
        actions = np.asarray(list(policy_dict.keys()), dtype=np.int64)
        probabilities = np.asarray(list(policy_dict.values()), dtype=np.float64)
        probability_sum = float(np.sum(probabilities))
        if probability_sum <= 0.0 or not np.isfinite(probability_sum):
            raise ValueError("Action policy must form a valid probability distribution.")
        probabilities /= probability_sum
        return int(np.random.choice(actions, p=probabilities))

    @staticmethod
    def _greedy_action(policy_dict: dict[int, float]) -> int:
        return max(policy_dict, key=policy_dict.get)

    @staticmethod
    def _normalize_policy_dict(policy_dict: dict[int, float]) -> dict[int, float]:
        if not policy_dict:
            return {}

        actions = np.asarray(list(policy_dict.keys()), dtype=np.int64)
        probabilities = np.asarray(list(policy_dict.values()), dtype=np.float64)
        valid_mask = np.isfinite(probabilities) & (probabilities > 0.0)
        if not np.any(valid_mask):
            return {}

        actions = actions[valid_mask]
        probabilities = probabilities[valid_mask]
        probability_sum = float(np.sum(probabilities))
        if probability_sum <= 0.0 or not np.isfinite(probability_sum):
            return {}

        probabilities /= probability_sum
        if not np.all(np.isfinite(probabilities)):
            return {}
        return {
            int(action): float(probability)
            for action, probability in zip(actions, probabilities)
        }

    def _fallback_uniform_policy(
        self,
        *,
        agent_index: int,
        policy_modes: tuple[str | None, str | None],
        root_node: SimultaneousMCTSNode,
    ) -> dict[int, float]:
        if agent_index == 0:
            candidate_actions = tuple(root_node.ego_available_actions)
        else:
            candidate_actions = tuple(root_node.npc_available_actions)

        if not candidate_actions:
            ego_actions, npc_actions = get_available_actions(
                self._require_env(),
                policy_modes=policy_modes,
            )
            candidate_actions = ego_actions if agent_index == 0 else npc_actions

        if not candidate_actions:
            raise RuntimeError(
                f"Could not resolve any valid actions for agent_index={agent_index}."
            )

        uniform_probability = 1.0 / len(candidate_actions)
        return {
            int(action): uniform_probability
            for action in candidate_actions
        }

    def _resolve_valid_policy(
        self,
        *,
        policy_dict: dict[int, float],
        agent_index: int,
        policy_modes: tuple[str | None, str | None],
        root_node: SimultaneousMCTSNode,
    ) -> dict[int, float]:
        normalized_policy = self._normalize_policy_dict(policy_dict)
        if normalized_policy:
            return normalized_policy

        fallback_policy = self._fallback_uniform_policy(
            agent_index=agent_index,
            policy_modes=policy_modes,
            root_node=root_node,
        )
        print(
            "[trainer] invalid/empty policy detected; "
            f"falling back to uniform distribution for agent_index={agent_index}",
            flush=True,
        )
        return fallback_policy

    def _root_matches_environment(
        self,
        *,
        root_node: SimultaneousMCTSNode,
        outcome: TerminalOutcome,
    ) -> bool:
        root_env = getattr(root_node, "env", None)
        if root_env is None:
            return False

        if root_node.terminal_outcome.terminal != outcome.terminal:
            return False

        try:
            current_snapshots = get_agent_snapshots(self._require_env())
            root_snapshots = get_agent_snapshots(root_env)
        except Exception:
            return False

        for current_snapshot, root_snapshot in zip(current_snapshots, root_snapshots):
            if (
                current_snapshot.on_road != root_snapshot.on_road
                or current_snapshot.crashed != root_snapshot.crashed
            ):
                return False
            if not np.allclose(
                current_snapshot.position,
                root_snapshot.position,
                atol=1e-3,
                rtol=0.0,
            ):
                return False
            if not np.isclose(
                current_snapshot.heading,
                root_snapshot.heading,
                atol=1e-3,
                rtol=0.0,
            ):
                return False
            if not np.isclose(
                current_snapshot.speed,
                root_snapshot.speed,
                atol=1e-3,
                rtol=0.0,
            ):
                return False

        if not np.isclose(
            get_progress_value(self._require_env()),
            get_progress_value(root_env),
            atol=1e-3,
            rtol=0.0,
        ):
            return False

        if outcome.terminal:
            return True
        return bool(root_node.ego_available_actions) and bool(root_node.npc_available_actions)

    def _finalize_outcome(
        self,
        *,
        terminal_outcome: TerminalOutcome,
        max_steps_reached: bool,
    ) -> TerminalOutcome:
        if terminal_outcome.terminal:
            return terminal_outcome
        if max_steps_reached:
            return TerminalOutcome(True, 0.0, 0.0, "max_steps_draw")
        return TerminalOutcome(True, 0.0, 0.0, "unfinished_draw")

    def _build_policy_target(
        self,
        policy_dict: dict[int, float],
        *,
        agent_index: int,
    ) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    def _make_episode_step_sample(
        self,
        *,
        state: np.ndarray,
        target_vector: np.ndarray,
        policy_dict: dict[int, float],
        agent_index: int,
    ) -> EpisodeStepSample:
        policy_targets = tuple(
            np.copy(policy_target)
            for policy_target in self._build_policy_target(
                policy_dict,
                agent_index=agent_index,
            )
        )
        return EpisodeStepSample(
            state=np.copy(state),
            target_vector=np.copy(target_vector),
            policy_targets=policy_targets,
            agent_index=int(agent_index),
        )

    def _make_replay_example(
        self,
        sample: EpisodeStepSample,
        outcome: TerminalOutcome,
    ):
        raise NotImplementedError

    def run_episode(
        self,
        *,
        seed: int = 21,
        episode_index: int = 0,
        max_steps: int | None = None,
        store_in_replay: bool = True,
        return_examples: bool = False,
        add_root_dirichlet_noise: bool | None = None,
        sample_actions: bool = True,
        step_callback=None,
    ) -> dict:
        env = self._require_env()
        np.random.seed(seed)
        env.reset(seed=seed)

        policy_modes = self._policy_modes_for_episode(episode_index)
        collect_npc_samples = self._should_collect_npc_samples(episode_index)
        root_node = self._build_root_node(policy_modes=policy_modes)
        mcts = self._build_mcts(root_node)
        if add_root_dirichlet_noise is None:
            add_root_dirichlet_noise = self.add_root_dirichlet_noise

        episode_samples: list[EpisodeStepSample] = []
        step_count = 0
        outcome = root_node.terminal_outcome

        while not outcome.terminal and (max_steps is None or step_count < max_steps):
            if not self._root_matches_environment(root_node=root_node, outcome=outcome):
                root_node = self._build_root_node(policy_modes=policy_modes)
                mcts = self._build_mcts(root_node)

            root_node.ensure_model_inputs(self.tensor_builder)
            state_batch = np.copy(root_node.cached_perspective_batch)
            target_vector_batch = np.copy(root_node.cached_target_vector_batch)

            mcts.prepare_root(add_exploration_noise=bool(add_root_dirichlet_noise))
            mcts.reset_timing_stats()
            self._run_rollouts(mcts)
            search_stats = mcts.get_timing_stats()

            temperature = self._temperature_for_step(step_count) if sample_actions else 0.0
            ego_policy = mcts.action_distribution(agent_index=0, temperature=temperature)
            npc_policy = mcts.action_distribution(
                agent_index=1,
                temperature=0.0 if policy_modes[1] is not None else temperature,
            )
            ego_policy = self._resolve_valid_policy(
                policy_dict=ego_policy,
                agent_index=0,
                policy_modes=policy_modes,
                root_node=root_node,
            )
            npc_policy = self._resolve_valid_policy(
                policy_dict=npc_policy,
                agent_index=1,
                policy_modes=policy_modes,
                root_node=root_node,
            )

            episode_samples.append(
                self._make_episode_step_sample(
                    state=state_batch[0],
                    target_vector=target_vector_batch[0],
                    policy_dict=ego_policy,
                    agent_index=0,
                )
            )
            if collect_npc_samples:
                episode_samples.append(
                    self._make_episode_step_sample(
                        state=state_batch[1],
                        target_vector=target_vector_batch[1],
                        policy_dict=npc_policy,
                        agent_index=1,
                    )
                )

            choose_action = self._sample_action if sample_actions and temperature > 1e-8 else self._greedy_action
            ego_action = choose_action(ego_policy)
            npc_action = choose_action(npc_policy) if policy_modes[1] is None else self._greedy_action(npc_policy)
            joint_action = (ego_action, npc_action)
            env.step(joint_action)
            step_count += 1
            post_step_outcome = classify_terminal_state(env, self.config.zero_sum)

            if step_callback is not None:
                step_callback(
                    {
                        "step": step_count,
                        "joint_action": joint_action,
                        "done": bool(post_step_outcome.terminal),
                        "outcome_reason": post_step_outcome.reason,
                        "search_stats": search_stats,
                    }
                )

            if self.reuse_tree_between_steps and joint_action in root_node.children:
                mcts.move_to_new_root(joint_action)
                root_node = mcts.root
                if not self._root_matches_environment(
                    root_node=root_node,
                    outcome=post_step_outcome,
                ):
                    root_node = self._build_root_node(policy_modes=policy_modes)
                    mcts = self._build_mcts(root_node)
            else:
                root_node = self._build_root_node(policy_modes=policy_modes)
                mcts = self._build_mcts(root_node)

            outcome = post_step_outcome

        max_steps_reached = max_steps is not None and step_count >= max_steps and not outcome.terminal
        outcome = self._finalize_outcome(
            terminal_outcome=outcome,
            max_steps_reached=max_steps_reached,
        )
        self.last_episode_outcome = outcome

        episode_examples = [
            self._make_replay_example(sample, outcome)
            for sample in episode_samples
        ]

        if store_in_replay:
            self.replay_buffer.extend(episode_examples)

        summary = {
            "seed": int(seed),
            "episode_index": int(episode_index),
            "steps": int(step_count),
            "outcome_reason": outcome.reason,
            "ego_value": float(outcome.ego_value),
            "npc_value": float(outcome.npc_value),
            "policy_modes": policy_modes,
            "collected_samples": len(episode_samples) if store_in_replay else 0,
        }
        if return_examples:
            summary["episode_examples"] = episode_examples
        self.last_episode_summary = summary
        if self.verbose:
            print(
                "episode="
                f"{summary['episode_index']} "
                f"steps={summary['steps']} "
                f"outcome={summary['outcome_reason']} "
                f"ego={summary['ego_value']:.2f} "
                f"npc={summary['npc_value']:.2f}"
            )
        return summary

    def fit(
        self,
        *,
        iterations: int,
        episodes_per_iteration: int,
        seed_start: int = 21,
        max_steps: int | None = None,
    ) -> list[dict]:
        training_summaries = []
        for iteration in range(int(iterations)):
            episode_summaries = []
            for episode_offset in range(int(episodes_per_iteration)):
                episode_index = iteration * int(episodes_per_iteration) + episode_offset
                episode_seed = seed_start + episode_index
                episode_summary = self.run_episode(
                    seed=episode_seed,
                    episode_index=episode_index,
                    max_steps=max_steps,
                    store_in_replay=True,
                    add_root_dirichlet_noise=True,
                    sample_actions=True,
                )
                episode_summaries.append(episode_summary)
            train_metrics = self.train()
            iteration_summary = {
                "iteration": iteration + 1,
                "episodes": episode_summaries,
                "train_metrics": train_metrics,
            }
            training_summaries.append(iteration_summary)
        return training_summaries

    def save_model(self, path: str | None = None) -> Path:
        output_path = Path(path or self.config.model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), output_path)
        return output_path

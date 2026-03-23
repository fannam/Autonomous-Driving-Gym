import copy
import time
from typing import Dict

import numpy as np
import torch

try:
    from core.policy import softmax_policy
    from core.settings import StackConfig
    from core.state_stack import init_state_stack, update_state_stack
except ModuleNotFoundError as exc:
    if exc.name != "core":
        raise
    from .policy import softmax_policy
    from .settings import StackConfig
    from .state_stack import init_state_stack, update_state_stack

TERMINAL_SUCCESS_VALUE = 1.0
TERMINAL_TIMEOUT_VALUE = 0.0
TERMINAL_FAILURE_VALUE = -1.0


def _record_timing(stats: dict | None, key: str, elapsed_s: float) -> None:
    if stats is not None:
        stats[key] = float(stats.get(key, 0.0)) + float(elapsed_s)


def _record_counter(stats: dict | None, key: str, increment: int = 1) -> None:
    if stats is not None:
        stats[key] = int(stats.get(key, 0)) + int(increment)


def _accumulate_env_step_profile(timing_stats: dict | None, env) -> None:
    if timing_stats is None:
        return

    env_unwrapped = getattr(env, "unwrapped", env)
    get_profile = getattr(env_unwrapped, "get_last_step_profile", None)
    if callable(get_profile):
        step_profile = get_profile() or {}
    else:
        step_profile = getattr(env_unwrapped, "_last_step_profile", None) or {}
    if not step_profile:
        return

    simulation_profile = step_profile.get("simulation") or {}
    _record_counter(timing_stats, "env_step_profile_count")
    _record_timing(
        timing_stats, "env_step_time_s", float(step_profile.get("step_time_s", 0.0))
    )
    _record_timing(
        timing_stats,
        "env_step_simulate_time_s",
        float(step_profile.get("simulate_time_s", 0.0)),
    )
    _record_timing(
        timing_stats,
        "env_step_observe_time_s",
        float(step_profile.get("observe_time_s", 0.0)),
    )
    _record_timing(
        timing_stats,
        "env_step_reward_time_s",
        float(step_profile.get("reward_time_s", 0.0)),
    )
    _record_timing(
        timing_stats,
        "env_step_terminated_time_s",
        float(step_profile.get("terminated_time_s", 0.0)),
    )
    _record_timing(
        timing_stats,
        "env_step_truncated_time_s",
        float(step_profile.get("truncated_time_s", 0.0)),
    )
    _record_timing(
        timing_stats,
        "env_step_info_time_s",
        float(step_profile.get("info_time_s", 0.0)),
    )
    _record_timing(
        timing_stats,
        "env_step_render_time_s",
        float(step_profile.get("render_time_s", 0.0)),
    )
    _record_counter(
        timing_stats,
        "env_simulation_frames_total",
        int(simulation_profile.get("frames", 0)),
    )
    _record_counter(
        timing_stats,
        "env_action_act_calls",
        int(simulation_profile.get("action_act_calls", 0)),
    )
    _record_timing(
        timing_stats,
        "env_frame_time_s",
        float(simulation_profile.get("frame_time_s", 0.0)),
    )
    _record_timing(
        timing_stats,
        "env_action_act_time_s",
        float(simulation_profile.get("action_act_time_s", 0.0)),
    )
    _record_timing(
        timing_stats,
        "env_road_act_time_s",
        float(simulation_profile.get("road_act_time_s", 0.0)),
    )
    _record_timing(
        timing_stats,
        "env_road_step_time_s",
        float(simulation_profile.get("road_step_time_s", 0.0)),
    )
    _record_timing(
        timing_stats,
        "env_auto_render_time_s",
        float(simulation_profile.get("automatic_render_time_s", 0.0)),
    )


def _resolve_mcts_device(device=None, use_cuda=False) -> torch.device:
    if device is not None:
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but CUDA is not available.")
        return resolved_device

    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def _collect_available_actions(env) -> tuple[int, ...]:
    """
    Return available discrete action ids for the current env state.

    - Prefer env-specific filtering if implemented.
    - Fallback to full discrete action space for DiscreteAction.
    """
    try:
        actions = env.unwrapped.get_available_actions()
        return tuple(int(action) for action in actions)
    except (AttributeError, NotImplementedError):
        pass

    action_space = env.action_space
    if hasattr(action_space, "n"):
        return tuple(range(int(action_space.n)))

    raise ValueError(
        "Current action space is not discrete. "
        "Please use a discretized action space (e.g. DiscreteAction)."
    )


class MCTSNode:
    def __init__(
        self,
        env,
        parent,
        parent_action,
        prior_prob,
        stack_config=None,
        copy_env=True,
        defer_stack_init=False,
    ):
        self.env = copy.deepcopy(env) if copy_env else env
        self.parent = parent
        self.parent_action = parent_action
        self.stack_config = stack_config or StackConfig()
        self.children: Dict[int, "MCTSNode"] = {}
        self._n = 0
        self._W = 0
        self._P = prior_prob

        ego_vehicle = self.env.unwrapped.road.vehicles[0]
        self.collision = bool(getattr(ego_vehicle, "crashed", False))
        success_fn = getattr(self.env.unwrapped, "_is_success", None)
        self.success = bool(success_fn()) if callable(success_fn) else False
        self.terminated = bool(self.env.unwrapped._is_terminated())
        self.truncated = bool(self.env.unwrapped._is_truncated())
        self.available_actions = (
            _collect_available_actions(self.env)
            if not self.success and not self.truncated and not self.terminated
            else tuple()
        )
        self.stack_of_planes = None

        if not defer_stack_init:
            self.ensure_stack_of_planes()

    def ensure_stack_of_planes(self, timing_stats: dict | None = None):
        if self.stack_of_planes is not None:
            return

        ensure_started_at = time.perf_counter()
        if self.parent is None:
            stack_init_started_at = time.perf_counter()
            parent_stack = init_state_stack(self.stack_config)
            _record_timing(
                timing_stats,
                "stack_init_time_s",
                time.perf_counter() - stack_init_started_at,
            )
        else:
            # Child nodes must not mutate the parent's cached stack.
            self.parent.ensure_stack_of_planes(timing_stats=timing_stats)
            parent_copy_started_at = time.perf_counter()
            parent_stack = self.parent.stack_of_planes.copy()
            _record_timing(
                timing_stats,
                "stack_parent_copy_time_s",
                time.perf_counter() - parent_copy_started_at,
            )

        observation_started_at = time.perf_counter()
        observation = self.env.unwrapped.observation_type.observe()
        _record_timing(
            timing_stats,
            "observation_time_s",
            time.perf_counter() - observation_started_at,
        )
        stack_update_started_at = time.perf_counter()
        self.stack_of_planes = update_state_stack(
            self.env,
            parent_stack,
            observation,
            stack_config=self.stack_config,
        )
        _record_timing(
            timing_stats,
            "stack_update_time_s",
            time.perf_counter() - stack_update_started_at,
        )
        _record_timing(
            timing_stats,
            "ensure_stack_time_s",
            time.perf_counter() - ensure_started_at,
        )

    def pucb_score(self, c_puct=2):
        q_value = 0 if self._n == 0 else self._W / self._n
        parent_visit_count = max(1, self.parent._n)
        exploration = c_puct * self._P * np.sqrt(parent_visit_count) / (1 + self._n)
        return q_value + exploration

    def select(self, c_puct=3):
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: child.pucb_score(c_puct))

    def expand(self, action_priors, timing_stats: dict | None = None):
        expand_started_at = time.perf_counter()
        children_created = 0
        for action in self.available_actions:
            prob = action_priors.get(action, 0.0)
            if action in self.children or prob <= 0:
                continue
            deepcopy_started_at = time.perf_counter()
            next_env = copy.deepcopy(self.env)
            _record_timing(
                timing_stats,
                "expand_deepcopy_time_s",
                time.perf_counter() - deepcopy_started_at,
            )
            env_step_started_at = time.perf_counter()
            next_env.step(action)
            _record_timing(
                timing_stats,
                "expand_env_step_time_s",
                time.perf_counter() - env_step_started_at,
            )
            _accumulate_env_step_profile(timing_stats, next_env)
            child_init_started_at = time.perf_counter()
            self.children[action] = MCTSNode(
                next_env,
                self,
                action,
                prob,
                stack_config=self.stack_config,
                copy_env=False,
                defer_stack_init=True,
            )
            _record_timing(
                timing_stats,
                "expand_node_init_time_s",
                time.perf_counter() - child_init_started_at,
            )
            children_created += 1
        _record_timing(
            timing_stats,
            "expand_time_s",
            time.perf_counter() - expand_started_at,
        )
        _record_counter(timing_stats, "expanded_children", children_created)

    def is_leaf(self):
        return self.children == {}

    def backpropagate(self, result):
        self._n += 1
        self._W += result

    def backpropagate_recursive(self, result):
        if self.parent:
            self.parent.backpropagate_recursive(result)
        self.backpropagate(result)


class MCTS:
    def __init__(
        self,
        root,
        network,
        use_cuda=False,
        device=None,
        c_puct=5,
        n_simulations=10,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
    ):
        self.c_puct = c_puct
        self.root = root
        self._device = _resolve_mcts_device(device=device, use_cuda=use_cuda)
        self._network = network.to(self._device)
        self._n_simulations = n_simulations
        self._root_dirichlet_alpha = float(root_dirichlet_alpha)
        self._root_exploration_fraction = float(root_exploration_fraction)
        self._pending_root_noise = False
        self.reset_timing_stats()
        if self._network.training:
            self._network.eval()

    def reset_timing_stats(self):
        self._timing_stats = {
            "rollouts": 0,
            "terminal_rollouts": 0,
            "inference_calls": 0,
            "expanded_children": 0,
            "rollout_time_s": 0.0,
            "inference_time_s": 0.0,
            "traverse_time_s": 0.0,
            "ensure_stack_time_s": 0.0,
            "stack_init_time_s": 0.0,
            "stack_parent_copy_time_s": 0.0,
            "observation_time_s": 0.0,
            "stack_update_time_s": 0.0,
            "tensor_prep_time_s": 0.0,
            "policy_dict_time_s": 0.0,
            "softmax_time_s": 0.0,
            "dirichlet_noise_time_s": 0.0,
            "expand_time_s": 0.0,
            "expand_deepcopy_time_s": 0.0,
            "expand_env_step_time_s": 0.0,
            "expand_node_init_time_s": 0.0,
            "env_step_profile_count": 0,
            "env_step_time_s": 0.0,
            "env_step_simulate_time_s": 0.0,
            "env_step_observe_time_s": 0.0,
            "env_step_reward_time_s": 0.0,
            "env_step_terminated_time_s": 0.0,
            "env_step_truncated_time_s": 0.0,
            "env_step_info_time_s": 0.0,
            "env_step_render_time_s": 0.0,
            "env_simulation_frames_total": 0,
            "env_frame_time_s": 0.0,
            "env_action_act_calls": 0,
            "env_action_act_time_s": 0.0,
            "env_road_act_time_s": 0.0,
            "env_road_step_time_s": 0.0,
            "env_auto_render_time_s": 0.0,
            "backprop_time_s": 0.0,
            "terminal_backprop_time_s": 0.0,
            "selection_depth_total": 0,
            "max_leaf_depth": 0,
        }

    def get_timing_stats(self):
        rollouts = int(self._timing_stats["rollouts"])
        terminal_rollouts = int(self._timing_stats["terminal_rollouts"])
        inference_calls = int(self._timing_stats["inference_calls"])
        expanded_children = int(self._timing_stats["expanded_children"])
        env_step_profile_count = int(self._timing_stats["env_step_profile_count"])
        env_simulation_frames_total = int(
            self._timing_stats["env_simulation_frames_total"]
        )
        rollout_time_s = float(self._timing_stats["rollout_time_s"])
        inference_time_s = float(self._timing_stats["inference_time_s"])
        env_step_time_s = float(self._timing_stats["env_step_time_s"])
        env_frame_time_s = float(self._timing_stats["env_frame_time_s"])
        avg_rollout_ms = 0.0 if rollouts == 0 else 1000.0 * rollout_time_s / rollouts
        avg_inference_ms = (
            0.0 if inference_calls == 0 else 1000.0 * inference_time_s / inference_calls
        )
        rollouts_per_sec = 0.0 if rollout_time_s <= 0.0 else rollouts / rollout_time_s
        return {
            "rollouts": rollouts,
            "terminal_rollouts": terminal_rollouts,
            "inference_calls": inference_calls,
            "expanded_children": expanded_children,
            "rollout_time_s": rollout_time_s,
            "inference_time_s": inference_time_s,
            "traverse_time_s": float(self._timing_stats["traverse_time_s"]),
            "ensure_stack_time_s": float(self._timing_stats["ensure_stack_time_s"]),
            "stack_init_time_s": float(self._timing_stats["stack_init_time_s"]),
            "stack_parent_copy_time_s": float(self._timing_stats["stack_parent_copy_time_s"]),
            "observation_time_s": float(self._timing_stats["observation_time_s"]),
            "stack_update_time_s": float(self._timing_stats["stack_update_time_s"]),
            "tensor_prep_time_s": float(self._timing_stats["tensor_prep_time_s"]),
            "policy_dict_time_s": float(self._timing_stats["policy_dict_time_s"]),
            "softmax_time_s": float(self._timing_stats["softmax_time_s"]),
            "dirichlet_noise_time_s": float(self._timing_stats["dirichlet_noise_time_s"]),
            "expand_time_s": float(self._timing_stats["expand_time_s"]),
            "expand_deepcopy_time_s": float(self._timing_stats["expand_deepcopy_time_s"]),
            "expand_env_step_time_s": float(self._timing_stats["expand_env_step_time_s"]),
            "expand_node_init_time_s": float(self._timing_stats["expand_node_init_time_s"]),
            "env_step_profile_count": env_step_profile_count,
            "env_step_time_s": env_step_time_s,
            "env_step_simulate_time_s": float(
                self._timing_stats["env_step_simulate_time_s"]
            ),
            "env_step_observe_time_s": float(
                self._timing_stats["env_step_observe_time_s"]
            ),
            "env_step_reward_time_s": float(self._timing_stats["env_step_reward_time_s"]),
            "env_step_terminated_time_s": float(
                self._timing_stats["env_step_terminated_time_s"]
            ),
            "env_step_truncated_time_s": float(
                self._timing_stats["env_step_truncated_time_s"]
            ),
            "env_step_info_time_s": float(self._timing_stats["env_step_info_time_s"]),
            "env_step_render_time_s": float(
                self._timing_stats["env_step_render_time_s"]
            ),
            "env_simulation_frames_total": env_simulation_frames_total,
            "env_frame_time_s": env_frame_time_s,
            "env_action_act_calls": int(self._timing_stats["env_action_act_calls"]),
            "env_action_act_time_s": float(self._timing_stats["env_action_act_time_s"]),
            "env_road_act_time_s": float(self._timing_stats["env_road_act_time_s"]),
            "env_road_step_time_s": float(self._timing_stats["env_road_step_time_s"]),
            "env_auto_render_time_s": float(self._timing_stats["env_auto_render_time_s"]),
            "backprop_time_s": float(self._timing_stats["backprop_time_s"]),
            "terminal_backprop_time_s": float(self._timing_stats["terminal_backprop_time_s"]),
            "selection_depth_total": int(self._timing_stats["selection_depth_total"]),
            "max_leaf_depth": int(self._timing_stats["max_leaf_depth"]),
            "avg_rollout_ms": avg_rollout_ms,
            "avg_inference_ms": avg_inference_ms,
            "rollouts_per_sec": rollouts_per_sec,
            "avg_env_step_ms": 0.0
            if env_step_profile_count == 0
            else 1000.0 * env_step_time_s / env_step_profile_count,
            "avg_env_frame_ms": 0.0
            if env_simulation_frames_total == 0
            else 1000.0 * env_frame_time_s / env_simulation_frames_total,
            "avg_expanded_children": 0.0 if rollouts == 0 else expanded_children / rollouts,
            "avg_leaf_depth": 0.0
            if rollouts == 0
            else float(self._timing_stats["selection_depth_total"]) / rollouts,
        }

    def traverse_to_leaf(self):
        node = self.root
        depth = 0
        while not node.is_leaf():
            node = node.select(self.c_puct)
            depth += 1
        return node, depth

    def _predict_policy_value(self, leaf_node):
        leaf_node.ensure_stack_of_planes(timing_stats=self._timing_stats)
        tensor_prep_started_at = time.perf_counter()
        state_tensor = torch.from_numpy(leaf_node.stack_of_planes).unsqueeze(0).to(
            device=self._device,
            non_blocking=self._device.type != "cpu",
        )
        _record_timing(
            self._timing_stats,
            "tensor_prep_time_s",
            time.perf_counter() - tensor_prep_started_at,
        )

        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        inference_started_at = time.perf_counter()
        with torch.inference_mode():
            predicted_policy, predicted_value = self._network(state_tensor)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        inference_elapsed = time.perf_counter() - inference_started_at
        self._timing_stats["inference_calls"] += 1
        self._timing_stats["inference_time_s"] += inference_elapsed

        policy_dict_started_at = time.perf_counter()
        policy_dict = {action: prob for action, prob in enumerate(predicted_policy.squeeze().tolist())}
        _record_timing(
            self._timing_stats,
            "policy_dict_time_s",
            time.perf_counter() - policy_dict_started_at,
        )
        return policy_dict, predicted_value.item()

    def _compute_rollout_value(self, truncated, terminated, crashed, success, predicted_value):
        if success:
            return TERMINAL_SUCCESS_VALUE
        if truncated:
            return TERMINAL_TIMEOUT_VALUE
        if terminated or crashed:
            return TERMINAL_FAILURE_VALUE
        return predicted_value

    def _apply_dirichlet_noise_to_policy(self, policy, available_actions):
        if not available_actions:
            return policy

        noise = np.random.dirichlet(
            [self._root_dirichlet_alpha] * len(available_actions)
        )
        mixed_policy = dict(policy)
        epsilon = self._root_exploration_fraction
        for action, noise_prob in zip(available_actions, noise):
            prior = float(policy.get(action, 0.0))
            mixed_policy[action] = (1.0 - epsilon) * prior + epsilon * float(noise_prob)
        return softmax_policy(mixed_policy, available_actions)

    def prepare_root(self, add_exploration_noise=False):
        self._pending_root_noise = bool(add_exploration_noise)
        if not self._pending_root_noise or not self.root.children:
            return

        current_policy = {
            action: child._P for action, child in self.root.children.items()
        }
        noised_policy = self._apply_dirichlet_noise_to_policy(
            current_policy,
            tuple(self.root.children.keys()),
        )
        for action, child in self.root.children.items():
            child._P = noised_policy[action]
        self._pending_root_noise = False

    def rollout(self):
        rollout_started_at = time.perf_counter()
        traverse_started_at = time.perf_counter()
        leaf_node, leaf_depth = self.traverse_to_leaf()
        _record_timing(
            self._timing_stats,
            "traverse_time_s",
            time.perf_counter() - traverse_started_at,
        )
        _record_counter(self._timing_stats, "selection_depth_total", leaf_depth)
        self._timing_stats["max_leaf_depth"] = max(
            int(self._timing_stats["max_leaf_depth"]),
            int(leaf_depth),
        )
        try:
            truncated = leaf_node.truncated
            terminated = leaf_node.terminated
            crashed = bool(leaf_node.collision)
            success = bool(leaf_node.success)

            if success or truncated or terminated:
                self._timing_stats["terminal_rollouts"] += 1
                rollout_value = self._compute_rollout_value(
                    truncated,
                    terminated,
                    crashed,
                    success,
                    predicted_value=0.0,
                )
                backprop_started_at = time.perf_counter()
                leaf_node.backpropagate_recursive(rollout_value)
                _record_timing(
                    self._timing_stats,
                    "terminal_backprop_time_s",
                    time.perf_counter() - backprop_started_at,
                )
                return

            predicted_policy, predicted_value = self._predict_policy_value(leaf_node)
            softmax_started_at = time.perf_counter()
            updated_policy = softmax_policy(predicted_policy, leaf_node.available_actions)
            _record_timing(
                self._timing_stats,
                "softmax_time_s",
                time.perf_counter() - softmax_started_at,
            )
            if leaf_node is self.root and self._pending_root_noise:
                root_noise_started_at = time.perf_counter()
                updated_policy = self._apply_dirichlet_noise_to_policy(
                    updated_policy,
                    leaf_node.available_actions,
                )
                _record_timing(
                    self._timing_stats,
                    "dirichlet_noise_time_s",
                    time.perf_counter() - root_noise_started_at,
                )
                self._pending_root_noise = False
            leaf_node.expand(updated_policy, timing_stats=self._timing_stats)

            rollout_value = self._compute_rollout_value(
                truncated,
                terminated,
                crashed,
                success,
                predicted_value,
            )
            backprop_started_at = time.perf_counter()
            leaf_node.backpropagate_recursive(rollout_value)
            _record_timing(
                self._timing_stats,
                "backprop_time_s",
                time.perf_counter() - backprop_started_at,
            )
        finally:
            self._timing_stats["rollouts"] += 1
            self._timing_stats["rollout_time_s"] += time.perf_counter() - rollout_started_at

    def move_to_new_root(self, action):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
            self._pending_root_noise = False
            return
        raise ValueError("Hanh dong khong co trong cay hien tai.")

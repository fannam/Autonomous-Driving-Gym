import copy
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

    def ensure_stack_of_planes(self):
        if self.stack_of_planes is not None:
            return

        if self.parent is None:
            parent_stack = init_state_stack(self.stack_config)
        else:
            # Child nodes must not mutate the parent's cached stack.
            self.parent.ensure_stack_of_planes()
            parent_stack = self.parent.stack_of_planes.copy()

        observation = self.env.unwrapped.observation_type.observe()
        self.stack_of_planes = update_state_stack(
            self.env,
            parent_stack,
            observation,
            stack_config=self.stack_config,
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

    def expand(self, action_priors):
        for action in self.available_actions:
            prob = action_priors.get(action, 0.0)
            if action in self.children or prob <= 0:
                continue
            next_env = copy.deepcopy(self.env)
            next_env.step(action)
            self.children[action] = MCTSNode(
                next_env,
                self,
                action,
                prob,
                stack_config=self.stack_config,
                copy_env=False,
                defer_stack_init=True,
            )

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
        if self._network.training:
            self._network.eval()

    def traverse_to_leaf(self):
        node = self.root
        while not node.is_leaf():
            node = node.select(self.c_puct)
        return node

    def _predict_policy_value(self, leaf_node):
        leaf_node.ensure_stack_of_planes()
        state_tensor = torch.from_numpy(leaf_node.stack_of_planes).unsqueeze(0).to(
            device=self._device,
            non_blocking=self._device.type != "cpu",
        )

        with torch.inference_mode():
            predicted_policy, predicted_value = self._network(state_tensor)

        policy_dict = {action: prob for action, prob in enumerate(predicted_policy.squeeze().tolist())}
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
        leaf_node = self.traverse_to_leaf()
        truncated = leaf_node.truncated
        terminated = leaf_node.terminated
        crashed = bool(leaf_node.collision)
        success = bool(leaf_node.success)

        if success or truncated or terminated:
            rollout_value = self._compute_rollout_value(
                truncated,
                terminated,
                crashed,
                success,
                predicted_value=0.0,
            )
            leaf_node.backpropagate_recursive(rollout_value)
            return

        predicted_policy, predicted_value = self._predict_policy_value(leaf_node)
        updated_policy = softmax_policy(predicted_policy, leaf_node.available_actions)
        if leaf_node is self.root and self._pending_root_noise:
            updated_policy = self._apply_dirichlet_noise_to_policy(
                updated_policy,
                leaf_node.available_actions,
            )
            self._pending_root_noise = False
        leaf_node.expand(updated_policy)

        rollout_value = self._compute_rollout_value(
            truncated,
            terminated,
            crashed,
            success,
            predicted_value,
        )
        leaf_node.backpropagate_recursive(rollout_value)

    def move_to_new_root(self, action):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
            self._pending_root_noise = False
            return
        raise ValueError("Hanh dong khong co trong cay hien tai.")

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

TERMINAL_TRUNCATED_VALUE = 1.0
TERMINAL_CRASHED_VALUE = -1.0


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
        if hasattr(ego_vehicle, "target_speeds"):
            min_speed = float(ego_vehicle.target_speeds[0])
            max_speed = float(ego_vehicle.target_speeds[-1])
        else:
            min_speed = float(getattr(ego_vehicle, "MIN_SPEED", -40.0))
            max_speed = float(getattr(ego_vehicle, "MAX_SPEED", 40.0))
        speed_span = max(max_speed - min_speed, 1e-6)
        self.speed_bonus = (float(ego_vehicle.speed) - min_speed) / speed_span
        self.collision = int(ego_vehicle.crashed)
        self.truncated = bool(self.env.unwrapped._is_truncated())
        self.available_actions = (
            _collect_available_actions(self.env)
            if not self.truncated and not self.collision
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
        exploration = c_puct * self._P * np.sqrt(np.log(self.parent._n) / (1 + self._n))
        speed_and_collision_bonus = 0.5 * self.speed_bonus - 0.5 * self.collision
        return q_value + exploration + speed_and_collision_bonus

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
    ):
        self.c_puct = c_puct
        self.root = root
        self._device = _resolve_mcts_device(device=device, use_cuda=use_cuda)
        self._network = network.to(self._device)
        self._n_simulations = n_simulations
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

    def _compute_rollout_value(self, truncated, crashed, predicted_value):
        if truncated:
            return TERMINAL_TRUNCATED_VALUE
        if crashed:
            return TERMINAL_CRASHED_VALUE
        return predicted_value

    def rollout(self):
        leaf_node = self.traverse_to_leaf()
        truncated = leaf_node.truncated
        crashed = bool(leaf_node.collision)

        if truncated or crashed:
            rollout_value = self._compute_rollout_value(truncated, crashed, predicted_value=0.0)
            leaf_node.backpropagate_recursive(rollout_value)
            return

        predicted_policy, predicted_value = self._predict_policy_value(leaf_node)
        updated_policy = softmax_policy(predicted_policy, leaf_node.available_actions)
        leaf_node.expand(updated_policy)

        rollout_value = self._compute_rollout_value(truncated, crashed, predicted_value)
        leaf_node.backpropagate_recursive(rollout_value)

    def move_to_new_root(self, action):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
            return
        raise ValueError("Hanh dong khong co trong cay hien tai.")

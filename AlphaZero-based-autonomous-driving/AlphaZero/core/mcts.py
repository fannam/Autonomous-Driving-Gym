import copy
from typing import Dict

import numpy as np
import torch

try:
    from core.policy import softmax_policy
    from core.state_stack import get_stack_of_grid, init_stack_of_grid
except ModuleNotFoundError as exc:
    if exc.name != "core":
        raise
    from .policy import softmax_policy
    from .state_stack import get_stack_of_grid, init_stack_of_grid

GRID_SIZE = (21, 5)
EGO_POSITION = (4, 2)
TERMINAL_TRUNCATED_VALUE = 1.0
TERMINAL_CRASHED_VALUE = -1.0


class MCTSNode:
    def __init__(self, env, parent, parent_action, prior_prob):
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.parent_action = parent_action
        self.children: Dict[int, "MCTSNode"] = {}
        self._n = 0
        self._W = 0
        self._P = prior_prob

        ego_vehicle = self.env.unwrapped.road.vehicles[0]
        min_speed = ego_vehicle.target_speeds[0]
        max_speed = ego_vehicle.target_speeds[-1]
        self.speed_bonus = (ego_vehicle.speed - min_speed) / (max_speed - min_speed)
        self.collision = 1 if ego_vehicle.crashed else 0

        if self.parent is None:
            self.stack_of_planes = init_stack_of_grid(GRID_SIZE, EGO_POSITION)
        else:
            observation = env.unwrapped.observation_type.observe()
            self.stack_of_planes = get_stack_of_grid(env, self.parent.stack_of_planes, observation)

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
        for action, prob in action_priors.items():
            if action not in self.children and prob > 0:
                next_env = copy.deepcopy(self.env)
                next_env.step(action)
                self.children[action] = MCTSNode(next_env, self, action, prob)

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
    def __init__(self, root, network, use_cuda=False, c_puct=5, n_simulations=10):
        self.c_puct = c_puct
        self.root = root
        self._network = network.cuda() if use_cuda else network
        self._n_simulations = n_simulations

    def traverse_to_leaf(self):
        node = self.root
        while not node.is_leaf():
            node = node.select(self.c_puct)
        return node

    def _predict_policy_value(self, leaf_node):
        state_tensor = torch.tensor(leaf_node.stack_of_planes, dtype=torch.float32).unsqueeze(0)
        predicted_policy, predicted_value = self._network(state_tensor)
        policy_dict = {action: prob for action, prob in enumerate(predicted_policy.squeeze().tolist())}
        return policy_dict, predicted_value.item()

    def _resolve_terminal_value(self, truncated, crashed, predicted_value):
        if truncated:
            return TERMINAL_TRUNCATED_VALUE
        if crashed:
            return TERMINAL_CRASHED_VALUE
        return predicted_value

    def rollout(self):
        leaf_node = self.traverse_to_leaf()
        truncated = leaf_node.env.unwrapped._is_truncated()
        crashed = leaf_node.env.unwrapped.road.vehicles[0].crashed

        predicted_policy, predicted_value = self._predict_policy_value(leaf_node)
        available_actions = leaf_node.env.unwrapped.get_available_actions()
        updated_policy = softmax_policy(predicted_policy, available_actions)

        if not truncated and not crashed:
            leaf_node.expand(updated_policy)

        rollout_value = self._resolve_terminal_value(truncated, crashed, predicted_value)
        leaf_node.backpropagate_recursive(rollout_value)

    def move_to_new_root(self, action):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
            return
        raise ValueError("Hanh dong khong co trong cay hien tai.")

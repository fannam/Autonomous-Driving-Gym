import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from core.mcts import MCTS, MCTSNode
    from core.settings import StackConfig
except ModuleNotFoundError as exc:
    if exc.name != "core":
        raise
    from ..core.mcts import MCTS, MCTSNode
    from ..core.settings import StackConfig


def _resolve_training_device(device) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but CUDA is not available.")
    return resolved_device


class AlphaZeroTrainer:
    def __init__(
        self,
        network,
        env,
        c_puct=2,
        n_simulations=10,
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        stack_config=None,
        n_actions=5,
        verbose=True,
        device="auto",
        max_root_visits=None,
    ):
        self.device = _resolve_training_device(device)
        self.network = network.to(self.device)
        self.env = env
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs
        self.stack_config = stack_config or StackConfig()
        self.n_actions = n_actions
        self.verbose = verbose
        self.max_root_visits = (
            None if max_root_visits is None else int(max_root_visits)
        )
        if self.max_root_visits is not None and self.max_root_visits <= 0:
            raise ValueError("max_root_visits must be a positive integer or None.")
        self.training_data = []
        self.action_list = []

    def _is_done(self):
        return self.env.unwrapped._is_truncated() or self.env.unwrapped._is_terminated()

    def _run_rollouts(self, mcts):
        rollout_budget = self.n_simulations
        if self.max_root_visits is not None:
            remaining_rollouts = self.max_root_visits - int(mcts.root._n)
            if remaining_rollouts <= 0:
                return 0
            rollout_budget = min(rollout_budget, remaining_rollouts)

        for _ in range(rollout_budget):
            mcts.rollout()
        return rollout_budget

    def _compute_action_probs(self, root_node):
        action_probs = {action: 0.0 for action in range(self.n_actions)}
        total_child_visits = sum(child._n for child in root_node.children.values())

        if total_child_visits <= 0:
            if root_node.children:
                uniform_prob = 1.0 / len(root_node.children)
                for action in root_node.children:
                    action_probs[action] = uniform_prob
                return action_probs

            available_actions = root_node.available_actions
            if available_actions:
                uniform_prob = 1.0 / len(available_actions)
                for action in available_actions:
                    if action in action_probs:
                        action_probs[action] = uniform_prob
                return action_probs

            uniform_prob = 1.0 / self.n_actions
            return {action: uniform_prob for action in range(self.n_actions)}

        for action, child in root_node.children.items():
            action_probs[action] = child._n / total_child_visits
        return action_probs

    def _estimate_root_value(self, root_node):
        return root_node._W / root_node._n if root_node._n > 0 else 0

    def self_play(self, seed=21, max_steps=None, step_callback=None):
        self.env.reset(seed=seed)
        done = self._is_done()
        step_count = 0

        root_node = MCTSNode(
            self.env,
            parent=None,
            parent_action=None,
            prior_prob=1.0,
            stack_config=self.stack_config,
        )
        mcts = MCTS(
            root=root_node,
            network=self.network,
            device=self.device,
            c_puct=self.c_puct,
            n_simulations=self.n_simulations,
        )

        while not done and (max_steps is None or step_count < max_steps):
            root_node.ensure_stack_of_planes()
            state = root_node.stack_of_planes
            self._run_rollouts(mcts)

            action_probs = self._compute_action_probs(root_node)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            predicted_value = self._estimate_root_value(root_node)

            self.training_data.append((state_tensor, action_probs, predicted_value))

            action = max(action_probs, key=action_probs.get)
            self.action_list.append(action)
            self.env.step(action)
            step_count += 1
            if self.verbose:
                print(f"action chosen: {action}")
            if step_callback is not None:
                step_callback(
                    {
                        "step": step_count,
                        "action": action,
                        "done": self._is_done(),
                    }
                )

            if action in root_node.children:
                mcts.move_to_new_root(action)
                root_node = mcts.root
            else:
                root_node = MCTSNode(
                    self.env,
                    parent=None,
                    parent_action=None,
                    prior_prob=1.0,
                    stack_config=self.stack_config,
                )
                mcts = MCTS(
                    root=root_node,
                    network=self.network,
                    device=self.device,
                    c_puct=self.c_puct,
                    n_simulations=self.n_simulations,
                )

            done = self._is_done()
        if self.verbose:
            print("end self-play")

    def _build_training_tensors(self):
        states, policies, values = zip(*self.training_data)
        state_tensor = torch.cat(states)
        policy_tensor = torch.tensor([list(policy.values()) for policy in policies], dtype=torch.float32)
        value_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        return state_tensor, policy_tensor, value_tensor

    def train(self):
        self.network.train()
        states, policies, values = self._build_training_tensors()
        dataset = TensorDataset(states, policies, values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for state_batch, policy_batch, value_batch in dataloader:
                state_batch = state_batch.to(self.device, non_blocking=self.device.type != "cpu")
                policy_batch = policy_batch.to(self.device, non_blocking=self.device.type != "cpu")
                value_batch = value_batch.to(self.device, non_blocking=self.device.type != "cpu")
                predicted_policy, predicted_value = self.network(state_batch)
                policy_loss = F.cross_entropy(predicted_policy, policy_batch)
                value_loss = F.mse_loss(predicted_value, value_batch)
                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}")

    def save_model(self, path="alphazero_model.pth"):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path="alphazero_model.pth"):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.to(self.device)

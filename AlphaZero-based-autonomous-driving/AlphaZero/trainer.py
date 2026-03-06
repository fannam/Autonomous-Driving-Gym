import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from MCTS import MCTS, MCTSNode
from stack_of_planes import get_stack_of_grid, init_stack_of_grid

GRID_SIZE = (21, 5)
EGO_POSITION = (4, 2)
N_ACTIONS = 5


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
    ):
        self.network = network
        self.env = env
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs
        self.training_data = []
        self.action_list = []

    def _is_done(self):
        return self.env.unwrapped._is_truncated() or self.env.unwrapped._is_terminated()

    def _run_rollouts(self, mcts):
        for _ in range(self.n_simulations):
            mcts.rollout()

    def _compute_action_probs(self, root_node):
        action_probs = {action: 0.0 for action in range(N_ACTIONS)}
        for action, child in root_node.children.items():
            action_probs[action] = child._n / (root_node._n - 1)
        return action_probs

    def _estimate_root_value(self, root_node):
        return root_node._W / root_node._n if root_node._n > 0 else 0

    def self_play(self, seed=21):
        """Generate training data with MCTS-based self-play."""
        self.env.reset(seed=seed)
        state = init_stack_of_grid(GRID_SIZE, EGO_POSITION)
        done = self._is_done()

        root_node = MCTSNode(self.env, parent=None, parent_action=None, prior_prob=1.0)
        mcts = MCTS(
            root=root_node,
            network=self.network,
            use_cuda=False,
            c_puct=self.c_puct,
            n_simulations=self.n_simulations,
        )

        while not done:
            state = get_stack_of_grid(state, self.env.unwrapped.observation_type.observe())
            self._run_rollouts(mcts)

            action_probs = self._compute_action_probs(root_node)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            predicted_value = self._estimate_root_value(root_node)

            self.training_data.append((state_tensor, action_probs, predicted_value))

            action = max(action_probs, key=action_probs.get)
            self.action_list.append(action)
            self.env.step(action)
            print(f"action chosen: {action}")

            if action in root_node.children:
                mcts.move_to_new_root(action)
                root_node = mcts.root
            else:
                raise ValueError("Action khong ton tai trong cay MCTS.")

            done = self._is_done()
        print("end self-play")

    def _build_training_tensors(self):
        states, policies, values = zip(*self.training_data)
        state_tensor = torch.cat(states)
        policy_tensor = torch.tensor([list(policy.values()) for policy in policies], dtype=torch.float32)
        value_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        return state_tensor, policy_tensor, value_tensor

    def train(self):
        """Train network with stored self-play data."""
        states, policies, values = self._build_training_tensors()
        dataset = TensorDataset(states, policies, values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for state_batch, policy_batch, value_batch in dataloader:
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
        self.network.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

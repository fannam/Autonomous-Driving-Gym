import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
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
        env=None,
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
        temperature=1.0,
        temperature_drop_step=None,
        add_root_dirichlet_noise=True,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        weight_decay=1e-4,
        reuse_tree_between_steps=True,
        max_expand_actions=None,
    ):
        self.device = _resolve_training_device(device)
        self.network = network.to(self.device)
        self.env = env
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
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
        self.temperature = float(temperature)
        self.temperature_drop_step = (
            None if temperature_drop_step is None else int(temperature_drop_step)
        )
        self.add_root_dirichlet_noise = bool(add_root_dirichlet_noise)
        self.root_dirichlet_alpha = float(root_dirichlet_alpha)
        self.root_exploration_fraction = float(root_exploration_fraction)
        self.reuse_tree_between_steps = bool(reuse_tree_between_steps)
        self.max_expand_actions = (
            None if max_expand_actions is None else int(max_expand_actions)
        )
        if self.max_expand_actions is not None and self.max_expand_actions <= 0:
            raise ValueError("max_expand_actions must be a positive integer or None.")
        self.training_data = []
        self.action_list = []
        self.last_episode_outcome = None

    @staticmethod
    def _soft_target_cross_entropy(policy_logits, policy_targets):
        normalized_targets = policy_targets / policy_targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
        log_policy = F.log_softmax(policy_logits, dim=1)
        return -(normalized_targets * log_policy).sum(dim=1).mean()

    def _require_env(self):
        if self.env is None:
            raise RuntimeError(
                "This AlphaZeroTrainer method requires an environment, but env=None was provided."
            )
        return self.env

    def _is_success(self):
        env = self._require_env()
        success_fn = getattr(env.unwrapped, "_is_success", None)
        return bool(success_fn()) if callable(success_fn) else False

    def _is_done(self):
        env = self._require_env()
        return (
            self._is_success()
            or env.unwrapped._is_truncated()
            or env.unwrapped._is_terminated()
        )

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

    def _build_mcts(self, root_node):
        return MCTS(
            root=root_node,
            network=self.network,
            device=self.device,
            c_puct=self.c_puct,
            n_simulations=self.n_simulations,
            root_dirichlet_alpha=self.root_dirichlet_alpha,
            root_exploration_fraction=self.root_exploration_fraction,
            max_expand_actions=self.max_expand_actions,
        )

    def _temperature_for_step(self, step_count):
        if self.temperature_drop_step is not None and step_count >= self.temperature_drop_step:
            return 0.0
        return self.temperature

    def _compute_action_probs(self, root_node, temperature):
        action_probs = {action: 0.0 for action in range(self.n_actions)}
        candidate_actions = (
            tuple(root_node.children.keys())
            if root_node.children
            else tuple(root_node.available_actions)
        )

        if not candidate_actions:
            uniform_prob = 1.0 / self.n_actions
            return {action: uniform_prob for action in range(self.n_actions)}

        visit_counts = np.asarray(
            [
                float(root_node.children[action]._n) if action in root_node.children else 0.0
                for action in candidate_actions
            ],
            dtype=np.float32,
        )
        total_child_visits = float(np.sum(visit_counts))
        if total_child_visits <= 0.0:
            uniform_prob = 1.0 / len(candidate_actions)
            for action in candidate_actions:
                action_probs[action] = uniform_prob
            return action_probs

        if temperature <= 1e-8:
            best_visit = np.max(visit_counts)
            best_actions = [
                action
                for action, visit_count in zip(candidate_actions, visit_counts)
                if visit_count == best_visit
            ]
            chosen_action = int(np.random.choice(best_actions))
            action_probs[chosen_action] = 1.0
            return action_probs

        scaled_counts = np.power(visit_counts, 1.0 / temperature).astype(
            np.float32,
            copy=False,
        )
        scaled_sum = float(np.sum(scaled_counts))
        if scaled_sum <= 0.0 or not np.isfinite(scaled_sum):
            uniform_prob = 1.0 / len(candidate_actions)
            for action in candidate_actions:
                action_probs[action] = uniform_prob
            return action_probs

        normalized_counts = scaled_counts / scaled_sum
        for action, prob in zip(candidate_actions, normalized_counts):
            action_probs[action] = float(prob)
        return action_probs

    @staticmethod
    def _sample_action(action_probs):
        actions = np.asarray(list(action_probs.keys()), dtype=np.int64)
        probabilities = np.asarray([action_probs[action] for action in actions], dtype=np.float64)
        prob_sum = float(np.sum(probabilities))
        if prob_sum <= 0.0 or not np.isfinite(prob_sum):
            raise ValueError("Action probabilities must form a valid distribution.")
        probabilities /= prob_sum
        return int(np.random.choice(actions, p=probabilities))

    def _compute_episode_outcome(self, episode_finished):
        env = self._require_env()
        if self._is_success():
            return 1.0
        if not episode_finished:
            return 0.0
        if env.unwrapped._is_truncated():
            return 0.0

        ego_vehicle = env.unwrapped.road.vehicles[0]
        if bool(getattr(ego_vehicle, "crashed", False)) or env.unwrapped._is_terminated():
            return -1.0
        return 0.0

    def search_policy(self, temperature=0.0, add_root_dirichlet_noise=False):
        env = self._require_env()
        root_node = MCTSNode(
            env,
            parent=None,
            parent_action=None,
            prior_prob=1.0,
            stack_config=self.stack_config,
        )
        mcts = self._build_mcts(root_node)
        mcts.prepare_root(add_exploration_noise=add_root_dirichlet_noise)
        self._run_rollouts(mcts)
        return self._compute_action_probs(root_node, temperature=temperature)

    def choose_action(self, temperature=0.0, add_root_dirichlet_noise=False, sample_from_policy=None):
        action_probs = self.search_policy(
            temperature=temperature,
            add_root_dirichlet_noise=add_root_dirichlet_noise,
        )
        if sample_from_policy is None:
            sample_from_policy = temperature > 1e-8

        if sample_from_policy:
            action = self._sample_action(action_probs)
        else:
            action = max(action_probs, key=action_probs.get)
        return action, action_probs

    def self_play(self, seed=21, max_steps=None, step_callback=None):
        env = self._require_env()
        np.random.seed(seed)
        env.reset(seed=seed)
        done = self._is_done()
        step_count = 0
        episode_history = []

        root_node = MCTSNode(
            env,
            parent=None,
            parent_action=None,
            prior_prob=1.0,
            stack_config=self.stack_config,
        )
        mcts = self._build_mcts(root_node)

        while not done and (max_steps is None or step_count < max_steps):
            root_prepare_started_at = time.perf_counter()
            root_node.ensure_stack_of_planes()
            state = root_node.stack_of_planes.copy()
            root_prepare_elapsed = time.perf_counter() - root_prepare_started_at
            mcts.prepare_root(add_exploration_noise=self.add_root_dirichlet_noise)
            mcts.reset_timing_stats()
            search_started_at = time.perf_counter()
            executed_rollouts = self._run_rollouts(mcts)
            search_elapsed = time.perf_counter() - search_started_at
            search_stats = mcts.get_timing_stats()
            search_stats["requested_rollouts"] = int(executed_rollouts)
            search_stats["search_time_s"] = float(search_elapsed)
            search_stats["root_prepare_time_s"] = float(root_prepare_elapsed)
            search_stats["search_overhead_s"] = max(
                0.0,
                float(search_elapsed) - float(search_stats["rollout_time_s"]),
            )
            search_stats["effective_rollouts_per_sec"] = (
                0.0
                if search_elapsed <= 0.0
                else float(search_stats["rollouts"]) / float(search_elapsed)
            )

            action_probs = self._compute_action_probs(
                root_node,
                temperature=self._temperature_for_step(step_count),
            )
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(dtype=torch.float32)
            episode_history.append((state_tensor, action_probs))

            action = self._sample_action(action_probs)
            self.action_list.append(action)
            env.step(action)
            step_count += 1
            if self.verbose:
                print(f"action chosen: {action}")
            if step_callback is not None:
                step_callback(
                    {
                        "step": step_count,
                        "action": action,
                        "done": self._is_done(),
                        "search_stats": search_stats,
                    }
                )

            if self.reuse_tree_between_steps and action in root_node.children:
                mcts.move_to_new_root(action)
                root_node = mcts.root
            else:
                root_node = MCTSNode(
                    env,
                    parent=None,
                    parent_action=None,
                    prior_prob=1.0,
                    stack_config=self.stack_config,
                )
                mcts = self._build_mcts(root_node)

            done = self._is_done()

        final_outcome = self._compute_episode_outcome(episode_finished=done)
        self.last_episode_outcome = final_outcome
        self.training_data.extend(
            (state_tensor, policy, final_outcome)
            for state_tensor, policy in episode_history
        )
        if self.verbose:
            print(f"end self-play outcome={final_outcome}")

    def _build_training_tensors(self):
        if not self.training_data:
            raise ValueError("No in-memory self-play samples are available for training.")

        states, policies, values = zip(*self.training_data)
        state_tensor = torch.cat(states)
        policy_tensor = torch.tensor(
            [[policy.get(action, 0.0) for action in range(self.n_actions)] for policy in policies],
            dtype=torch.float32,
        )
        value_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        return state_tensor, policy_tensor, value_tensor

    @staticmethod
    def _coerce_training_tensors(states, policies, values):
        state_tensor = torch.as_tensor(states, dtype=torch.float32)
        policy_tensor = torch.as_tensor(policies, dtype=torch.float32)
        value_tensor = torch.as_tensor(values, dtype=torch.float32)

        if state_tensor.ndim != 4:
            raise ValueError(
                f"Expected states with shape (batch, channels, width, height), got {tuple(state_tensor.shape)}."
            )
        if policy_tensor.ndim != 2:
            raise ValueError(
                f"Expected policies with shape (batch, n_actions), got {tuple(policy_tensor.shape)}."
            )
        if value_tensor.ndim == 1:
            value_tensor = value_tensor.unsqueeze(1)
        elif value_tensor.ndim != 2 or value_tensor.shape[1] != 1:
            raise ValueError(
                f"Expected values with shape (batch, 1), got {tuple(value_tensor.shape)}."
            )

        batch_size = state_tensor.shape[0]
        if batch_size == 0:
            raise ValueError("Training tensors are empty.")
        if policy_tensor.shape[0] != batch_size or value_tensor.shape[0] != batch_size:
            raise ValueError(
                "States, policies, and values must contain the same number of samples."
            )
        return state_tensor, policy_tensor, value_tensor

    def _train_from_dataset(self, dataset, shuffle=True):
        self.network.train()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        epoch_metrics = []

        for epoch in range(self.epochs):
            epoch_loss_sum = 0.0
            epoch_policy_loss_sum = 0.0
            epoch_value_loss_sum = 0.0
            sample_count = 0

            for state_batch, policy_batch, value_batch in dataloader:
                state_batch = state_batch.to(self.device, non_blocking=self.device.type != "cpu")
                policy_batch = policy_batch.to(self.device, non_blocking=self.device.type != "cpu")
                value_batch = value_batch.to(self.device, non_blocking=self.device.type != "cpu")
                policy_logits, predicted_value = self.network(state_batch, return_logits=True)
                policy_loss = self._soft_target_cross_entropy(policy_logits, policy_batch)
                value_loss = F.mse_loss(predicted_value, value_batch)
                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
                self.optimizer.step()

                batch_items = int(state_batch.shape[0])
                sample_count += batch_items
                epoch_loss_sum += float(loss.detach().item()) * batch_items
                epoch_policy_loss_sum += float(policy_loss.detach().item()) * batch_items
                epoch_value_loss_sum += float(value_loss.detach().item()) * batch_items

            epoch_metric = {
                "epoch": epoch + 1,
                "loss": epoch_loss_sum / sample_count,
                "policy_loss": epoch_policy_loss_sum / sample_count,
                "value_loss": epoch_value_loss_sum / sample_count,
            }
            epoch_metrics.append(epoch_metric)
            if self.verbose:
                print(
                    "Epoch "
                    f"{epoch_metric['epoch']}/{self.epochs}, "
                    f"loss={epoch_metric['loss']:.6f}, "
                    f"policy_loss={epoch_metric['policy_loss']:.6f}, "
                    f"value_loss={epoch_metric['value_loss']:.6f}"
                )

        return epoch_metrics

    def train_from_tensors(self, states, policies, values, shuffle=True):
        state_tensor, policy_tensor, value_tensor = self._coerce_training_tensors(
            states,
            policies,
            values,
        )
        dataset = TensorDataset(state_tensor, policy_tensor, value_tensor)
        return self._train_from_dataset(dataset, shuffle=shuffle)

    def train(self):
        states, policies, values = self._build_training_tensors()
        return self.train_from_tensors(states, policies, values, shuffle=True)

    def save_model(self, path="alphazero_model.pth"):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), output_path)

    def load_model(self, path="alphazero_model.pth"):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.to(self.device)

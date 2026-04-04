from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from autonomous_driving_shared.alphazero_adversarial.training.base import (
    BaseAdversarialAlphaZeroTrainer,
    EpisodeStepSample,
)

from ..core.mcts import SimultaneousMCTS, SimultaneousMCTSNode
from ..core.policy import policy_dict_to_array


class AdversarialAlphaZeroTrainer(BaseAdversarialAlphaZeroTrainer):
    def _build_mcts(self, root_node: SimultaneousMCTSNode) -> SimultaneousMCTS:
        return SimultaneousMCTS(
            root=root_node,
            network=self.network,
            tensor_builder=self.tensor_builder,
            device=self.device,
            c_puct=self.config.c_puct,
            n_simulations=self.config.n_simulations,
            root_dirichlet_alpha=self.config.root_dirichlet_alpha,
            root_exploration_fraction=self.config.root_exploration_fraction,
            max_expand_actions_per_agent=self.config.max_expand_actions_per_agent,
            relative_pruning_gamma=self.config.relative_pruning_gamma,
            discount_gamma=self.config.discount_gamma,
        )

    def _build_policy_target(
        self,
        policy_dict: dict[int, float],
        *,
        agent_index: int,
    ) -> tuple[np.ndarray, ...]:
        del agent_index
        return (policy_dict_to_array(policy_dict, self.config.n_actions),)

    def _make_replay_example(
        self,
        sample: EpisodeStepSample,
        *,
        value_target: float,
    ):
        (policy_vector,) = sample.policy_targets
        return (
            sample.state,
            sample.target_vector,
            policy_vector,
            float(value_target),
        )

    def _build_training_tensors(self):
        if not self.replay_buffer:
            raise ValueError("Replay buffer is empty; collect self-play data first.")
        states, target_vectors, policies, values = zip(*self.replay_buffer)
        state_tensor = torch.as_tensor(np.stack(states, axis=0), dtype=torch.float32)
        target_vector_tensor = torch.as_tensor(
            np.stack(target_vectors, axis=0),
            dtype=torch.float32,
        )
        policy_tensor = torch.as_tensor(np.stack(policies, axis=0), dtype=torch.float32)
        value_tensor = torch.as_tensor(np.asarray(values, dtype=np.float32)).unsqueeze(1)
        return state_tensor, target_vector_tensor, policy_tensor, value_tensor

    def train(self):
        state_tensor, target_vector_tensor, policy_tensor, value_tensor = self._build_training_tensors()
        dataset = TensorDataset(
            state_tensor,
            target_vector_tensor,
            policy_tensor,
            value_tensor,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=int(self.config.batch_size),
            shuffle=True,
        )

        self.network.train()
        epoch_metrics = []
        for epoch in range(int(self.config.epochs)):
            epoch_loss_sum = 0.0
            epoch_policy_loss_sum = 0.0
            epoch_value_loss_sum = 0.0
            sample_count = 0

            for state_batch, target_vector_batch, policy_batch, value_batch in dataloader:
                state_batch = state_batch.to(self.device, non_blocking=self.device.type != "cpu")
                target_vector_batch = target_vector_batch.to(
                    self.device,
                    non_blocking=self.device.type != "cpu",
                )
                policy_batch = policy_batch.to(self.device, non_blocking=self.device.type != "cpu")
                value_batch = value_batch.to(self.device, non_blocking=self.device.type != "cpu")

                policy_logits, predicted_value = self.network(
                    state_batch,
                    target_vector_batch,
                    return_logits=True,
                )
                policy_loss = self._kl_policy_loss(policy_logits, policy_batch)
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

            metrics = {
                "epoch": epoch + 1,
                "loss": epoch_loss_sum / sample_count,
                "policy_loss": epoch_policy_loss_sum / sample_count,
                "value_loss": epoch_value_loss_sum / sample_count,
            }
            epoch_metrics.append(metrics)
            if self.verbose:
                print(
                    "train "
                    f"epoch={metrics['epoch']} "
                    f"loss={metrics['loss']:.6f} "
                    f"policy={metrics['policy_loss']:.6f} "
                    f"value={metrics['value_loss']:.6f}"
                )

        self.network.eval()
        return epoch_metrics

    def load_model(self, path: str | None = None) -> None:
        model_path = Path(path or self.config.model_path)
        state_dict = torch.load(model_path, map_location=self.device)
        try:
            self.network.load_state_dict(state_dict)
        except RuntimeError as exc:
            if self.config.target_vector_dim > 0:
                raise RuntimeError(
                    f"Could not load checkpoint from {model_path}. "
                    "The current meta-adversarial network expects the late-fusion "
                    "target-vector layers, so checkpoints created before this "
                    "upgrade are not architecture-compatible."
                ) from exc
            raise
        self.network.to(self.device)
        self.network.eval()

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
from ..core.policy import marginalize_action_policy, policy_dict_to_array


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
            n_action_axis_0=self.config.n_action_axis_0,
            n_action_axis_1=self.config.n_action_axis_1,
            relative_pruning_gamma=self.config.relative_pruning_gamma,
            flip_npc_steering=self.config.tensor.flip_npc_perspective,
        )

    @staticmethod
    def _normalize_axis_target(target: np.ndarray) -> np.ndarray:
        normalized = np.asarray(target, dtype=np.float32).reshape(-1)
        target_sum = float(np.sum(normalized))
        if target_sum <= 0.0 or not np.isfinite(target_sum):
            if normalized.shape[0] == 0:
                return normalized
            normalized = np.full_like(normalized, 1.0 / normalized.shape[0])
            return normalized
        normalized /= target_sum
        return normalized

    def _maybe_smooth_axis_target(self, target: np.ndarray) -> np.ndarray:
        normalized = self._normalize_axis_target(target)
        if (
            not bool(self.config.use_policy_target_smoothing)
            or normalized.shape[0] <= 1
        ):
            return normalized

        sigma = float(self.config.policy_target_smoothing_sigma)
        axis_positions = np.arange(normalized.shape[0], dtype=np.float32)
        squared_distance = (axis_positions[:, None] - axis_positions[None, :]) ** 2
        smoothing_kernel = np.exp(-0.5 * squared_distance / (sigma**2)).astype(np.float32)
        smoothed = smoothing_kernel @ normalized
        return self._normalize_axis_target(smoothed)

    def _policy_dict_to_factorized_targets(
        self,
        policy_dict: dict[int, float],
        *,
        agent_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        flat_policy = policy_dict_to_array(policy_dict, self.config.n_actions)
        accelerate_target, steering_target = marginalize_action_policy(
            flat_policy,
            n_action_axis_0=self.config.n_action_axis_0,
            n_action_axis_1=self.config.n_action_axis_1,
            flip_steering=bool(
                agent_index == 1 and self.config.tensor.flip_npc_perspective
            ),
        )
        return (
            self._maybe_smooth_axis_target(accelerate_target),
            self._maybe_smooth_axis_target(steering_target),
        )

    def _build_policy_target(
        self,
        policy_dict: dict[int, float],
        *,
        agent_index: int,
    ) -> tuple[np.ndarray, ...]:
        accelerate_target, steering_target = self._policy_dict_to_factorized_targets(
            policy_dict,
            agent_index=agent_index,
        )
        return accelerate_target, steering_target

    def _make_replay_example(
        self,
        sample: EpisodeStepSample,
        outcome,
    ):
        accelerate_target, steering_target = sample.policy_targets
        value = float(
            outcome.ego_value if sample.agent_index == 0 else outcome.npc_value
        )
        return (
            sample.state,
            sample.target_vector,
            accelerate_target,
            steering_target,
            value,
        )

    def _build_training_tensors(self):
        if not self.replay_buffer:
            raise ValueError("Replay buffer is empty; collect self-play data first.")
        (
            states,
            target_vectors,
            accelerate_policies,
            steering_policies,
            values,
        ) = zip(*self.replay_buffer)
        state_tensor = torch.as_tensor(np.stack(states, axis=0), dtype=torch.float32)
        target_vector_tensor = torch.as_tensor(
            np.stack(target_vectors, axis=0),
            dtype=torch.float32,
        )
        accelerate_policy_tensor = torch.as_tensor(
            np.stack(accelerate_policies, axis=0),
            dtype=torch.float32,
        )
        steering_policy_tensor = torch.as_tensor(
            np.stack(steering_policies, axis=0),
            dtype=torch.float32,
        )
        value_tensor = torch.as_tensor(np.asarray(values, dtype=np.float32)).unsqueeze(1)
        return (
            state_tensor,
            target_vector_tensor,
            accelerate_policy_tensor,
            steering_policy_tensor,
            value_tensor,
        )

    def train(self):
        (
            state_tensor,
            target_vector_tensor,
            accelerate_policy_tensor,
            steering_policy_tensor,
            value_tensor,
        ) = self._build_training_tensors()
        dataset = TensorDataset(
            state_tensor,
            target_vector_tensor,
            accelerate_policy_tensor,
            steering_policy_tensor,
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
            epoch_accelerate_policy_loss_sum = 0.0
            epoch_steering_policy_loss_sum = 0.0
            epoch_value_loss_sum = 0.0
            sample_count = 0

            for (
                state_batch,
                target_vector_batch,
                accelerate_policy_batch,
                steering_policy_batch,
                value_batch,
            ) in dataloader:
                state_batch = state_batch.to(self.device, non_blocking=self.device.type != "cpu")
                target_vector_batch = target_vector_batch.to(
                    self.device,
                    non_blocking=self.device.type != "cpu",
                )
                accelerate_policy_batch = accelerate_policy_batch.to(
                    self.device,
                    non_blocking=self.device.type != "cpu",
                )
                steering_policy_batch = steering_policy_batch.to(
                    self.device,
                    non_blocking=self.device.type != "cpu",
                )
                value_batch = value_batch.to(self.device, non_blocking=self.device.type != "cpu")

                accelerate_logits, steering_logits, predicted_value = self.network(
                    state_batch,
                    target_vector_batch,
                    return_logits=True,
                )
                accelerate_policy_loss = self._kl_policy_loss(
                    accelerate_logits,
                    accelerate_policy_batch,
                )
                steering_policy_loss = self._kl_policy_loss(
                    steering_logits,
                    steering_policy_batch,
                )
                policy_loss = accelerate_policy_loss + steering_policy_loss
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
                epoch_accelerate_policy_loss_sum += (
                    float(accelerate_policy_loss.detach().item()) * batch_items
                )
                epoch_steering_policy_loss_sum += (
                    float(steering_policy_loss.detach().item()) * batch_items
                )
                epoch_value_loss_sum += float(value_loss.detach().item()) * batch_items

            metrics = {
                "epoch": epoch + 1,
                "loss": epoch_loss_sum / sample_count,
                "policy_loss": epoch_policy_loss_sum / sample_count,
                "accelerate_policy_loss": epoch_accelerate_policy_loss_sum / sample_count,
                "steering_policy_loss": epoch_steering_policy_loss_sum / sample_count,
                "value_loss": epoch_value_loss_sum / sample_count,
            }
            epoch_metrics.append(metrics)
            if self.verbose:
                print(
                    "train "
                    f"epoch={metrics['epoch']} "
                    f"loss={metrics['loss']:.6f} "
                    f"policy={metrics['policy_loss']:.6f} "
                    f"accel={metrics['accelerate_policy_loss']:.6f} "
                    f"steer={metrics['steering_policy_loss']:.6f} "
                    f"value={metrics['value_loss']:.6f}"
                )

        self.network.eval()
        return epoch_metrics

    def load_model(self, path: str | None = None) -> None:
        model_path = Path(path or self.config.model_path)
        self.network.load_state_dict(torch.load(model_path, map_location=self.device))
        self.network.to(self.device)
        self.network.eval()

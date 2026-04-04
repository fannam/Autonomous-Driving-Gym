from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from autonomous_driving_shared.alphazero_adversarial.network.base import (
    BaseAlphaZeroNetwork,
)


class AlphaZeroNetwork(BaseAlphaZeroNetwork):
    def __init__(
        self,
        input_shape,
        n_residual_layers=10,
        n_actions=25,
        n_action_axis_0=None,
        n_action_axis_1=None,
        channels=256,
        dropout_p=0.1,
        target_vector_dim=0,
        target_hidden_dim=32,
    ):
        action_count = int(n_actions)
        action_axis_0 = None if n_action_axis_0 is None else int(n_action_axis_0)
        action_axis_1 = None if n_action_axis_1 is None else int(n_action_axis_1)
        if action_axis_0 is None and action_axis_1 is None:
            inferred_axis = int(round(float(action_count) ** 0.5))
            if inferred_axis * inferred_axis != action_count:
                raise ValueError(
                    "n_actions is not a perfect square; "
                    "provide n_action_axis_0 and n_action_axis_1 explicitly."
                )
            action_axis_0 = inferred_axis
            action_axis_1 = inferred_axis
        elif action_axis_0 is None:
            if action_count % action_axis_1 != 0:
                raise ValueError(
                    "n_actions must be divisible by n_action_axis_1 when inferring axis 0."
                )
            action_axis_0 = action_count // action_axis_1
        elif action_axis_1 is None:
            if action_count % action_axis_0 != 0:
                raise ValueError(
                    "n_actions must be divisible by n_action_axis_0 when inferring axis 1."
                )
            action_axis_1 = action_count // action_axis_0

        if action_axis_0 * action_axis_1 != action_count:
            raise ValueError(
                "Factorized action axes do not match n_actions: "
                f"{action_axis_0} * {action_axis_1} != {action_count}."
            )

        super().__init__(
            input_shape,
            n_residual_layers=n_residual_layers,
            channels=channels,
            dropout_p=dropout_p,
            target_vector_dim=target_vector_dim,
            target_hidden_dim=target_hidden_dim,
        )
        self.n_actions = action_count
        self.n_action_axis_0 = action_axis_0
        self.n_action_axis_1 = action_axis_1

        self.shared_head_conv = nn.Conv2d(self.channels, 4, kernel_size=1)
        self.shared_head_bn = nn.BatchNorm2d(4)
        self.spatial_fc = nn.Linear(self.board_area * 4, 1024)
        self.spatial_dropout = nn.Dropout(self.dropout_p)

        self.fusion_fc = nn.Linear(self.fusion_input_dim(1024), 1024)
        self.fusion_dropout = nn.Dropout(self.dropout_p)
        self.accelerate_fc = nn.Linear(1024, self.n_action_axis_0)
        self.steering_fc = nn.Linear(1024, self.n_action_axis_1)
        self.value_fc = nn.Linear(1024, 1)

        self._initialize_weights()

    def forward_spatial_embedding(self, features):
        spatial = self.activation(self.shared_head_bn(self.shared_head_conv(features)))
        spatial = spatial.flatten(start_dim=1)
        spatial = self.activation(self.spatial_fc(spatial))
        spatial = self.spatial_dropout(spatial)
        return spatial

    def forward_heads(self, spatial_embedding, target_embedding):
        fused = self.fuse_embeddings(
            spatial_embedding,
            target_embedding,
            fusion_fc=self.fusion_fc,
            fusion_dropout=self.fusion_dropout,
        )
        accelerate_logits = self.accelerate_fc(fused)
        steering_logits = self.steering_fc(fused)
        value = torch.tanh(self.value_fc(fused))
        return accelerate_logits, steering_logits, value

    def forward(self, x, target_vector=None, return_logits=False):
        features = self.forward_features(x)
        spatial_embedding = self.forward_spatial_embedding(features)
        target_embedding = self.forward_target_embedding(
            target_vector,
            batch_size=int(spatial_embedding.shape[0]),
            device=spatial_embedding.device,
            dtype=spatial_embedding.dtype,
        )
        accelerate_logits, steering_logits, value = self.forward_heads(
            spatial_embedding,
            target_embedding,
        )
        if return_logits:
            return accelerate_logits, steering_logits, value
        return (
            F.softmax(accelerate_logits, dim=1),
            F.softmax(steering_logits, dim=1),
            value,
        )

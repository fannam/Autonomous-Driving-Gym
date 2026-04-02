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
        n_actions=5,
        channels=256,
        dropout_p=0.1,
        target_vector_dim=0,
        target_hidden_dim=32,
    ):
        super().__init__(
            input_shape,
            n_residual_layers=n_residual_layers,
            channels=channels,
            dropout_p=dropout_p,
            target_vector_dim=target_vector_dim,
            target_hidden_dim=target_hidden_dim,
        )
        self.n_actions = int(n_actions)

        self.spatial_conv = nn.Conv2d(self.channels, 4, kernel_size=1)
        self.spatial_bn = nn.BatchNorm2d(4)
        self.spatial_fc = nn.Linear(self.board_area * 4, 256)
        self.spatial_dropout = nn.Dropout(self.dropout_p)

        self.fusion_fc = nn.Linear(self.fusion_input_dim(256), 256)
        self.fusion_dropout = nn.Dropout(self.dropout_p)
        self.policy_fc = nn.Linear(256, self.n_actions)
        self.value_fc = nn.Linear(256, 1)

        self._initialize_weights()

    def forward_spatial_embedding(self, features):
        spatial = self.activation(self.spatial_bn(self.spatial_conv(features)))
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
        policy_logits = self.policy_fc(fused)
        value = torch.tanh(self.value_fc(fused))
        return policy_logits, value

    def forward(self, x, target_vector=None, return_logits=False):
        features = self.forward_features(x)
        spatial_embedding = self.forward_spatial_embedding(features)
        target_embedding = self.forward_target_embedding(
            target_vector,
            batch_size=int(spatial_embedding.shape[0]),
            device=spatial_embedding.device,
            dtype=spatial_embedding.dtype,
        )
        policy_logits, value = self.forward_heads(spatial_embedding, target_embedding)
        if return_logits:
            return policy_logits, value
        return F.softmax(policy_logits, dim=1), value

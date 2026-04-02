from __future__ import annotations

import torch
import torch.nn as nn


class BaseAlphaZeroNetwork(nn.Module):
    def __init__(
        self,
        input_shape,
        *,
        n_residual_layers: int = 10,
        channels: int = 256,
        dropout_p: float = 0.1,
        target_vector_dim: int = 0,
        target_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input_shape=(width, height, channels), got {input_shape!r}."
            )

        self.input_shape = input_shape
        self.channels = int(channels)
        self.dropout_p = float(dropout_p)
        self.board_area = int(input_shape[0] * input_shape[1])
        self.target_vector_dim = int(target_vector_dim)
        self.target_hidden_dim = int(target_hidden_dim)
        self.activation = nn.SiLU(inplace=True)

        self.conv_layer = nn.Conv2d(input_shape[2], self.channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(self.channels)
        self.residual_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.channels),
                )
                for _ in range(n_residual_layers)
            ]
        )

        if self.target_vector_dim > 0:
            self.target_fc1 = nn.Linear(self.target_vector_dim, self.target_hidden_dim)
            self.target_fc2 = nn.Linear(self.target_hidden_dim, self.target_hidden_dim)
        else:
            self.target_fc1 = None
            self.target_fc2 = None

    def fusion_input_dim(self, spatial_embedding_dim: int) -> int:
        if self.target_vector_dim <= 0:
            return int(spatial_embedding_dim)
        return int(spatial_embedding_dim) + self.target_hidden_dim

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for residual in self.residual_layers:
            nn.init.zeros_(residual[-1].weight)

    def forward_features(self, x):
        x = self.activation(self.batch_norm(self.conv_layer(x)))
        for residual in self.residual_layers:
            residual_x = x
            x = residual(x) + residual_x
            x = self.activation(x)
        return x

    def forward_target_embedding(self, target_vector, *, batch_size, device, dtype):
        if self.target_vector_dim <= 0:
            return None
        if target_vector is None:
            target_vector = torch.zeros(
                (batch_size, self.target_vector_dim),
                device=device,
                dtype=dtype,
            )
        target = self.activation(self.target_fc1(target_vector))
        target = self.activation(self.target_fc2(target))
        return target

    def fuse_embeddings(self, spatial_embedding, target_embedding, *, fusion_fc, fusion_dropout):
        fused = spatial_embedding
        if target_embedding is not None:
            fused = torch.cat((spatial_embedding, target_embedding), dim=1)
        fused = self.activation(fusion_fc(fused))
        fused = fusion_dropout(fused)
        return fused

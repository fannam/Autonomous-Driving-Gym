from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.activation(x + residual)
        return x


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        n_actions: int,
        stem_channels: int = 32,
        downsample_channels: int = 64,
        latent_dim: int = 256,
        residual_blocks: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.n_actions = int(n_actions)
        self.stem_channels = int(stem_channels)
        self.downsample_channels = int(downsample_channels)
        self.latent_dim = int(latent_dim)
        self.residual_blocks = int(residual_blocks)
        self.dropout_p = float(dropout_p)
        self.activation = nn.SiLU(inplace=True)

        blocks_before = max(1, self.residual_blocks // 2)
        blocks_after = max(1, self.residual_blocks - blocks_before)

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, self.stem_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.stem_channels),
            nn.SiLU(inplace=True),
        )
        self.residual_stack1 = nn.Sequential(
            *[ResidualBlock(self.stem_channels) for _ in range(blocks_before)]
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(
                self.stem_channels,
                self.downsample_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.downsample_channels),
            nn.SiLU(inplace=True),
        )
        self.residual_stack2 = nn.Sequential(
            *[ResidualBlock(self.downsample_channels) for _ in range(blocks_after)]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.downsample_channels, self.latent_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.policy_head = nn.Linear(self.latent_dim, self.n_actions)
        self.value_head = nn.Linear(self.latent_dim, 1)

        self._initialize_weights()

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

    def forward_features(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.stem(obs)
        x = self.residual_stack1(x)
        x = self.downsample(x)
        x = self.residual_stack2(x)
        x = self.pool(x).flatten(start_dim=1)
        x = self.activation(self.fc(x))
        x = self.dropout(x)
        return x

    def forward(
        self,
        obs: torch.Tensor,
        *,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(obs)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        if return_logits:
            return policy_logits, value
        return F.softmax(policy_logits, dim=1), value


def build_actor_critic(config) -> ActorCriticNetwork:
    return ActorCriticNetwork(
        input_channels=int(config.input_channels),
        n_actions=int(config.n_actions),
        stem_channels=int(config.network.stem_channels),
        downsample_channels=int(config.network.downsample_channels),
        latent_dim=int(config.network.latent_dim),
        residual_blocks=int(config.network.residual_blocks),
        dropout_p=float(config.network.dropout_p),
    )


__all__ = ["ActorCriticNetwork", "build_actor_critic"]

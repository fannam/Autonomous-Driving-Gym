from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.GroupNorm(
            _resolve_group_count(int(out_channels)),
            int(out_channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        stride: int = 1,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        resolved_out_channels = int(out_channels or in_channels)
        self.conv1 = ConvNormAct(
            int(in_channels),
            resolved_out_channels,
            kernel_size=3,
            stride=int(stride),
        )
        self.conv2 = nn.Conv2d(
            resolved_out_channels,
            resolved_out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(
            _resolve_group_count(resolved_out_channels),
            resolved_out_channels,
        )
        self.dropout = (
            nn.Dropout2d(float(dropout_p))
            if float(dropout_p) > 0.0
            else nn.Identity()
        )
        if int(stride) != 1 or int(in_channels) != resolved_out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    int(in_channels),
                    resolved_out_channels,
                    kernel_size=1,
                    stride=int(stride),
                    bias=False,
                ),
                nn.GroupNorm(
                    _resolve_group_count(resolved_out_channels),
                    resolved_out_channels,
                ),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        n_actions: int,
        stem_channels: int = 48,
        downsample_channels: int = 96,
        latent_dim: int = 384,
        residual_blocks: int = 10,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.n_actions = int(n_actions)
        self.stem_channels = int(stem_channels)
        self.downsample_channels = int(downsample_channels)
        self.latent_dim = int(latent_dim)
        self.residual_blocks = int(residual_blocks)
        self.dropout_p = float(dropout_p)
        self.stage3_channels = max(
            self.downsample_channels * 2,
            self.latent_dim // 2,
        )

        stage1_blocks, stage2_blocks, stage3_blocks = self._split_stage_blocks(
            self.residual_blocks
        )
        block_dropout = min(0.2, self.dropout_p * 0.5)

        self.stem = nn.Sequential(
            ConvNormAct(
                self.input_channels,
                self.stem_channels,
                kernel_size=5,
                padding=2,
            ),
            ConvNormAct(
                self.stem_channels,
                self.stem_channels,
                kernel_size=3,
            ),
        )
        self.stage1 = nn.Sequential(
            *[
                ResidualBlock(
                    self.stem_channels,
                    dropout_p=block_dropout,
                )
                for _ in range(stage1_blocks)
            ]
        )
        self.transition1 = ResidualBlock(
            self.stem_channels,
            self.downsample_channels,
            stride=2,
            dropout_p=block_dropout,
        )
        self.stage2 = nn.Sequential(
            *[
                ResidualBlock(
                    self.downsample_channels,
                    dropout_p=block_dropout,
                )
                for _ in range(stage2_blocks)
            ]
        )
        self.transition2 = ResidualBlock(
            self.downsample_channels,
            self.stage3_channels,
            stride=2,
            dropout_p=block_dropout,
        )
        self.stage3 = nn.Sequential(
            *[
                ResidualBlock(
                    self.stage3_channels,
                    dropout_p=block_dropout,
                )
                for _ in range(stage3_blocks)
            ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(self.stage3_channels * 2, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(self.dropout_p),
        )
        self.policy_head = nn.Linear(self.latent_dim, self.n_actions)
        self.value_head = nn.Linear(self.latent_dim, 1)

        self._initialize_weights()

    @staticmethod
    def _split_stage_blocks(total_blocks: int) -> tuple[int, int, int]:
        counts = [0, 0, 0]
        for index in range(max(0, int(total_blocks))):
            counts[index % 3] += 1
        return counts[0], counts[1], counts[2]

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.stem(obs)
        x = self.stage1(x)
        x = self.transition1(x)
        x = self.stage2(x)
        x = self.transition2(x)
        x = self.stage3(x)
        avg_features = self.avg_pool(x).flatten(start_dim=1)
        max_features = self.max_pool(x).flatten(start_dim=1)
        merged_features = torch.cat((avg_features, max_features), dim=1)
        return self.projection(merged_features)

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

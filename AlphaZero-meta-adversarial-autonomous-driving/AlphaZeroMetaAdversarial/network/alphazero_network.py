import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroNetwork(nn.Module):
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
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input_shape=(width, height, channels), got {input_shape!r}."
            )

        self.input_shape = input_shape
        self.n_actions = int(n_actions)
        self.channels = int(channels)
        self.dropout_p = float(dropout_p)
        self.board_area = int(input_shape[0] * input_shape[1])
        self.target_vector_dim = int(target_vector_dim)
        self.target_hidden_dim = int(target_hidden_dim)
        self.activation = nn.SiLU(inplace=True)

        self.conv_layer = nn.Conv2d(
            input_shape[2], self.channels, kernel_size=3, padding=1
        )
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

        self.spatial_conv = nn.Conv2d(self.channels, 4, kernel_size=1)
        self.spatial_bn = nn.BatchNorm2d(4)
        self.spatial_fc = nn.Linear(self.board_area * 4, 256)
        self.spatial_dropout = nn.Dropout(self.dropout_p)

        if self.target_vector_dim > 0:
            self.target_fc1 = nn.Linear(self.target_vector_dim, self.target_hidden_dim)
            self.target_fc2 = nn.Linear(self.target_hidden_dim, self.target_hidden_dim)
            fusion_input_dim = 256 + self.target_hidden_dim
        else:
            self.target_fc1 = None
            self.target_fc2 = None
            fusion_input_dim = 256

        self.fusion_fc = nn.Linear(fusion_input_dim, 256)
        self.fusion_dropout = nn.Dropout(self.dropout_p)
        self.policy_fc = nn.Linear(256, self.n_actions)
        self.value_fc = nn.Linear(256, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
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

    def forward_spatial_embedding(self, features):
        spatial = self.activation(self.spatial_bn(self.spatial_conv(features)))
        spatial = torch.flatten(spatial, start_dim=1)
        spatial = self.activation(self.spatial_fc(spatial))
        spatial = self.spatial_dropout(spatial)
        return spatial

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

    def forward_heads(self, spatial_embedding, target_embedding):
        fused = spatial_embedding
        if target_embedding is not None:
            fused = torch.cat((spatial_embedding, target_embedding), dim=1)
        fused = self.activation(self.fusion_fc(fused))
        fused = self.fusion_dropout(fused)
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

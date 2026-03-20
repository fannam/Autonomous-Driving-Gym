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
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input_shape=(width, height, channels), got {input_shape!r}."
            )

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.channels = int(channels)
        self.dropout_p = float(dropout_p)
        self.board_area = int(input_shape[0] * input_shape[1])
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

        self.value_conv = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.fc_input_size = self.board_area
        self.value_fc1 = nn.Linear(self.fc_input_size, 256)
        self.value_dropout = nn.Dropout(self.dropout_p)
        self.value_fc2 = nn.Linear(256, 1)

        self.policy_conv = nn.Conv2d(self.channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(self.fc_input_size * 2, n_actions)
        self.policy_dropout = nn.Dropout(self.dropout_p)

        self._initialize_weights()

    def _initialize_weights(self):
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

        # Start residual branches close to identity for stabler early training.
        for residual in self.residual_layers:
            last_batch_norm = residual[-1]
            nn.init.zeros_(last_batch_norm.weight)

    def forward_features(self, x):
        x = self.activation(self.batch_norm(self.conv_layer(x)))

        for residual in self.residual_layers:
            residual_x = x
            x = residual(x) + residual_x
            x = self.activation(x)
        return x

    def forward_heads(self, features):
        value = self.activation(self.value_bn(self.value_conv(features)))
        value = torch.flatten(value, start_dim=1)
        value = self.activation(self.value_fc1(value))
        value = self.value_dropout(value)
        value = torch.tanh(self.value_fc2(value))

        policy = self.activation(self.policy_bn(self.policy_conv(features)))
        policy = torch.flatten(policy, start_dim=1)
        policy = self.policy_dropout(policy)
        policy_logits = self.policy_fc(policy)
        return policy_logits, value

    def forward(self, x, return_logits=False):
        features = self.forward_features(x)
        policy_logits, value = self.forward_heads(features)

        if return_logits:
            return policy_logits, value

        policy = F.softmax(policy_logits, dim=1)
        return policy, value

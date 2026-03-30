import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroNetwork(nn.Module):
    def __init__(
        self,
        input_shape,
        n_residual_layers=10,
        n_actions=25,
        n_action_axis_0=None,
        n_action_axis_1=None,
        channels=256,
        dropout_p=0.1,
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input_shape=(width, height, channels), got {input_shape!r}."
            )

        resolved_n_actions = int(n_actions)
        resolved_axis_0 = None if n_action_axis_0 is None else int(n_action_axis_0)
        resolved_axis_1 = None if n_action_axis_1 is None else int(n_action_axis_1)
        if resolved_axis_0 is None and resolved_axis_1 is None:
            inferred_axis = int(round(float(resolved_n_actions) ** 0.5))
            if inferred_axis * inferred_axis != resolved_n_actions:
                raise ValueError(
                    "n_actions is not a perfect square; "
                    "provide n_action_axis_0 and n_action_axis_1 explicitly."
                )
            resolved_axis_0 = inferred_axis
            resolved_axis_1 = inferred_axis
        elif resolved_axis_0 is None:
            if resolved_n_actions % resolved_axis_1 != 0:
                raise ValueError(
                    "n_actions must be divisible by n_action_axis_1 when inferring axis 0."
                )
            resolved_axis_0 = resolved_n_actions // resolved_axis_1
        elif resolved_axis_1 is None:
            if resolved_n_actions % resolved_axis_0 != 0:
                raise ValueError(
                    "n_actions must be divisible by n_action_axis_0 when inferring axis 1."
                )
            resolved_axis_1 = resolved_n_actions // resolved_axis_0

        if resolved_axis_0 * resolved_axis_1 != resolved_n_actions:
            raise ValueError(
                "Factorized action axes do not match n_actions: "
                f"{resolved_axis_0} * {resolved_axis_1} != {resolved_n_actions}."
            )

        self.input_shape = input_shape
        self.n_actions = resolved_n_actions
        self.n_action_axis_0 = resolved_axis_0
        self.n_action_axis_1 = resolved_axis_1
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

        self.shared_head_conv = nn.Conv2d(self.channels, 4, kernel_size=1)
        self.shared_head_bn = nn.BatchNorm2d(4)
        self.shared_head_fc = nn.Linear(self.board_area * 4, 1024)
        self.shared_head_dropout = nn.Dropout(self.dropout_p)

        self.accelerate_fc = nn.Linear(1024, self.n_action_axis_0)
        self.steering_fc = nn.Linear(1024, self.n_action_axis_1)
        self.value_fc = nn.Linear(1024, 1)

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

        for residual in self.residual_layers:
            nn.init.zeros_(residual[-1].weight)

    def forward_features(self, x):
        x = self.activation(self.batch_norm(self.conv_layer(x)))
        for residual in self.residual_layers:
            residual_x = x
            x = residual(x) + residual_x
            x = self.activation(x)
        return x

    def forward_heads(self, features):
        shared = self.activation(self.shared_head_bn(self.shared_head_conv(features)))
        shared = torch.flatten(shared, start_dim=1)
        shared = self.activation(self.shared_head_fc(shared))
        shared = self.shared_head_dropout(shared)

        accelerate_logits = self.accelerate_fc(shared)
        steering_logits = self.steering_fc(shared)
        value = torch.tanh(self.value_fc(shared))
        return accelerate_logits, steering_logits, value

    def forward(self, x, return_logits=False):
        features = self.forward_features(x)
        accelerate_logits, steering_logits, value = self.forward_heads(features)
        if return_logits:
            return accelerate_logits, steering_logits, value
        return (
            F.softmax(accelerate_logits, dim=1),
            F.softmax(steering_logits, dim=1),
            value,
        )

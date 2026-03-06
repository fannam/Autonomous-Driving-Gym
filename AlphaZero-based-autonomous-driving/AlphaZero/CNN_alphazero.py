import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNetwork(nn.Module):
    def __init__(self, input_shape, n_residual_layers=10, n_actions=5):
        super(AlphaZeroNetwork, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        # Convolution đầu tiên
        self.conv_layer = nn.Conv2d(input_shape[2], 256, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(256)

        # Residual layers
        self.residual_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256)
            )
            for _ in range(n_residual_layers)
        ])

        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.fc_input_size = input_shape[0] * input_shape[1]

        self.value_fc1 = nn.Linear(self.fc_input_size, 256)
        self.value_dropout = nn.Dropout(0.5)  # Thêm Dropout
        self.value_fc2 = nn.Linear(256, 1)

        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(self.fc_input_size * 2, n_actions)
        self.policy_dropout = nn.Dropout(0.5)  # Thêm Dropout

    def forward(self, x):
        # Convolution đầu tiên và batch normalization
        x = F.relu(self.batch_norm(self.conv_layer(x)))

        # Residual layers
        for residual in self.residual_layers:
            residual_x = x
            x = residual(x) + residual_x
            x = F.relu(x)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = self.value_dropout(value)  # Áp dụng Dropout
        value = torch.tanh(self.value_fc2(value))

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(policy.size(0), -1)
        policy = self.policy_fc(self.policy_dropout(policy))  # Áp dụng Dropout
        policy = F.softmax(policy, dim=1)

        return policy, value

import torch
import torch.nn as nn
from torch import Tensor


class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.temperature = nn.parameter.Parameter(torch.Tensor([1.0]))

        self.alpha_mean = nn.parameter.Parameter(torch.Tensor([1.0]))

        self.alpha_cov = nn.parameter.Parameter(torch.Tensor([1.0]))

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
        )

        self.mean_layer = nn.Sequential(nn.Linear(64, action_dim))

        self.std_layer = nn.Sequential(nn.Linear(64, action_dim))

    def forward(
        self, states: Tensor
    ) -> tuple[tuple[Tensor, Tensor], Tensor, tuple[Tensor, Tensor]]:
        x = self.nn_layer(states)
        mean = self.mean_layer(x[:, :64])
        std = self.std_layer(x[:, 64:]).exp()

        return (mean, std), self.temperature, (self.alpha_mean, self.alpha_cov)


class Value_Model(nn.Module):
    def __init__(self, state_dim):
        super(Value_Model, self).__init__()

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.nn_layer(states)

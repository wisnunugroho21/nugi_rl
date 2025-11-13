import torch
import torch.nn as nn
from torch import Tensor


class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Policy_Model, self).__init__()

        self.action_dim = action_dim

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
        )

        self.mean_layer = nn.Sequential(nn.Linear(64, action_dim), nn.Tanh())

        self.std_layer = nn.Sequential(nn.Linear(64, action_dim), nn.Sigmoid())

    def forward(self, states: Tensor) -> Tensor:
        x = self.nn_layer(states)
        mean = self.mean_layer(x[:, :64])
        std = self.std_layer(x[:, 64:])

        return torch.stack([mean, std])


class Value_Model(nn.Module):
    def __init__(self, state_dim: int):
        super(Value_Model, self).__init__()

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.nn_layer(states)

import torch
import torch.nn as nn
from torch import Tensor


class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.actor_std = nn.parameter.Parameter(torch.zeros(action_dim))

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
        )

        self.mean_layer = nn.Sequential(nn.Linear(128, action_dim), nn.Tanh())

    def forward(self, states: Tensor) -> Tensor:
        x = self.nn_layer(states)
        mean = self.mean_layer(x)
        std = self.actor_std.exp()

        return torch.stack([mean, std])


class Q_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_Model, self).__init__()

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, states, actions: Tensor) -> Tensor:
        x = torch.cat((states, actions), -1)

        return self.nn_layer(x)

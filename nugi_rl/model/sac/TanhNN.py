import torch
import torch.nn as nn
from torch import Tensor


class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.actor_std = nn.parameter.Parameter(torch.zeros(action_dim))

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        self.mean_layer = nn.Sequential(nn.Linear(64, action_dim), nn.Tanh())

    def forward(self, states: Tensor) -> tuple:
        x = self.nn_layer(states)
        mean = self.mean_layer(x)
        std = self.actor_std.exp()

        return (mean, std)


class Q_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_Model, self).__init__()

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, states, actions: Tensor) -> tuple:
        x = torch.cat((states, actions), -1)

        return self.nn_layer(x)

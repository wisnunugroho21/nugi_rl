import torch
import torch.nn as nn
from torch import Tensor


class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, init_alpha=1.0):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.mean_layer = nn.Sequential(nn.Linear(64, action_dim), nn.Tanh())

        self.std_layer = nn.Sequential(nn.Linear(64, action_dim), nn.Sigmoid())

        self.alpha = nn.parameter.Parameter(torch.tensor([init_alpha]))

    def forward(self, states: Tensor) -> tuple:
        x = self.nn_layer(states)
        mean = self.mean_layer(x[:, :64])
        std = self.std_layer(x[:, 64:])

        return (mean, std), self.alpha


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

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        x = torch.cat((states, actions), -1)

        return self.nn_layer(x)

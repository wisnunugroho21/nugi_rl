import torch
import torch.nn as nn
from torch import Tensor

# from nugi_rl.model.components.ReLUKAN import ReLUKAN


class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, init_alpha=1.0):
        super(Policy_Model, self).__init__()

        self.action_dim = action_dim

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2 * action_dim),
        )

        self.alpha = nn.parameter.Parameter(torch.tensor([init_alpha]))

        # self.nn_layer = nn.Sequential(
        #     ReLUKAN(state_dim, 32), ReLUKAN(32, 8), ReLUKAN(8, 2 * action_dim)
        # )

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        x = self.nn_layer(states)
        mean = torch.tanh(x[:, : self.action_dim])
        std = torch.sigmoid(x[:, self.action_dim :])

        return torch.stack([mean, std]), self.alpha


class Q_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_Model, self).__init__()

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # self.nn_layer = nn.Sequential(
        #     ReLUKAN(state_dim + action_dim, 32),
        #     ReLUKAN(32, 8),
        #     ReLUKAN(8, 1),
        # )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        x = torch.cat((states, actions), -1)

        return self.nn_layer(x)

import torch
import torch.nn as nn
from torch import Tensor

from nugi_rl.model.components.ReluKan import HighOrderReLUKAN


class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
            HighOrderReLUKAN(state_dim, (2 * state_dim + 1) * action_dim, order=4),
            HighOrderReLUKAN(
                (2 * state_dim + 1) * action_dim,
                (2 * state_dim + 1) * action_dim,
                order=4,
            ),
            HighOrderReLUKAN((2 * state_dim + 1) * action_dim, action_dim, order=4),
            nn.Tanh(),
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.nn_layer(states)


class Q_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_Model, self).__init__()

        in_dim = state_dim + action_dim

        self.nn_layer = nn.Sequential(
            HighOrderReLUKAN(in_dim, (2 * in_dim + 1), order=4),
            HighOrderReLUKAN((2 * in_dim + 1), (2 * in_dim + 1), order=4),
            HighOrderReLUKAN((2 * in_dim + 1), 1, order=4),
        )

    def forward(self, states, actions: Tensor) -> Tensor:
        x = torch.cat((states, actions), -1)
        return self.nn_layer(x)

import torch.nn as nn
from torch import Tensor

from nugi_rl.model.components.ReLUKAN import ReLUKAN


class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Policy_Model, self).__init__()

        mid_dim = (state_dim + action_dim) * 2 + 1

        self.nn_layer = nn.Sequential(
            ReLUKAN(state_dim, mid_dim),
            ReLUKAN(mid_dim, action_dim),
            nn.Softmax(-1),
        )

    def forward(self, states: Tensor) -> Tensor:
        action_datas = self.nn_layer(states)
        return action_datas


class Value_Model(nn.Module):
    def __init__(self, state_dim: int):
        super(Value_Model, self).__init__()

        mid_dim = state_dim * 2 + 1

        self.nn_layer = nn.Sequential(
            ReLUKAN(state_dim, mid_dim),
            ReLUKAN(mid_dim, 1),
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.nn_layer(states)

import torch.nn as nn
from torch import Tensor

class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 32),
          nn.ReLU(),
          nn.Linear(32, 32),
          nn.ReLU(),
          nn.Linear(32, action_dim),
          nn.Softmax(-1)
        )
        
    def forward(self, states: Tensor, detach: bool = False) -> Tensor:
      action_datas = self.nn_layer(states).detach()

      if detach:
        return (action_datas.detach())
      else:
        return (action_datas)

class Value_Model(nn.Module):
    def __init__(self, state_dim: int):
        super(Value_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 32),
          nn.ReLU(),
          nn.Linear(32, 32),
          nn.ReLU(),
          nn.Linear(32, 1)
        )
        
    def forward(self, states: Tensor, detach: bool = False) -> Tensor:
      action = self.nn_layer(states)

      if detach:
        return (action.detach(), )
      else:
        return (action, )
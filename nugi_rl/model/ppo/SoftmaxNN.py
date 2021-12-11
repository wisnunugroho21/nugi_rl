import torch.nn as nn
from torch import Tensor

class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 256),
          nn.SiLU(),
          nn.Linear(256, 128),
          nn.SiLU(),
          nn.Linear(128, action_dim),
          nn.Softmax(-1)
        )
        
    def forward(self, states: Tensor, detach: bool = False) -> tuple:
      action_datas = self.nn_layer(states).detach()

      if detach:
        return (action_datas.detach(), )
      else:
        return (action_datas, )

class Value_Model(nn.Module):
    def __init__(self, state_dim: int):
        super(Value_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU(),
          nn.Linear(64, 1)
        )
        
    def forward(self, states: Tensor, detach: bool = False) -> Tensor:
      if detach:
        return self.nn_layer(states).detach()
      else:
        return self.nn_layer(states)
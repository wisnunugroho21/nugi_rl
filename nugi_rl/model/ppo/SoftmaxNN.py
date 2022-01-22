import torch.nn as nn
from torch import Tensor

class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 640),
          nn.ReLU(),
          nn.Linear(640, 640),
          nn.ReLU(),
          nn.Linear(640, action_dim),
          nn.Softmax(-1)
        )
        
    def forward(self, states: Tensor) -> tuple:
      action_datas = self.nn_layer(states).detach()
      return (action_datas.detach(), )

class Value_Model(nn.Module):
    def __init__(self, state_dim: int):
        super(Value_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 640),
          nn.ReLU(),
          nn.Linear(640, 640),
          nn.ReLU(),
          nn.Linear(640, 1)
        )
        
    def forward(self, states: Tensor) -> Tensor:
      return self.nn_layer(states)
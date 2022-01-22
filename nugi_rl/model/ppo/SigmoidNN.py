import torch.nn as nn
from torch import Tensor

class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, bins: int):
        super(Policy_Model, self).__init__()

        self.action_dim = action_dim
        self.bins = bins

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 128),
          nn.SiLU(),
          nn.Linear(128, 256),
          nn.SiLU(),
          nn.Linear(256, 64),
          nn.SiLU(),
          nn.Linear(64, action_dim * bins),
          nn.Sigmoid()
        )
        
    def forward(self, states: Tensor) -> tuple:
      action = self.nn_layer(states)
      action = action.reshape(-1, self.action_dim, self.bins)

      return (action.detach(), )

class Value_Model(nn.Module):
    def __init__(self, state_dim: int):
        super(Value_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 64),
          nn.SiLU(),
          nn.Linear(64, 128),
          nn.SiLU(),
          nn.Linear(128, 32),
          nn.SiLU(),
          nn.Linear(32, 1)
        )
        
    def forward(self, states: Tensor) -> Tensor:
      return self.nn_layer(states)
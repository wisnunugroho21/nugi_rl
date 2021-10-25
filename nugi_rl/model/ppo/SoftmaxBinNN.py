import torch.nn as nn
from torch import Tensor

class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, bins: int):
        super(Policy_Model, self).__init__()

        self.action_dim = action_dim
        self.bins = bins

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU(),
          nn.Linear(128, action_dim * bins)
        )
        
    def forward(self, states: Tensor, detach: bool = False) -> Tensor:
      action = self.nn_layer(states)
      action = action.reshape(-1, self.action_dim, self.bins)
      action = nn.functional.softmax(action, dim = -1)

      if detach:
        return action.detach()
      else:
        return action

class Value_Model(nn.Module):
    def __init__(self, state_dim: int):
        super(Value_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 64),
          nn.ReLU(),
          nn.Linear(64, 1)
        )
        
    def forward(self, states: Tensor, detach: bool = False) -> Tensor:
      if detach:
        return self.nn_layer(states).detach()
      else:
        return self.nn_layer(states)
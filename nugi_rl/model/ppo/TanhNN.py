import torch
import torch.nn as nn
from torch import Tensor

class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Policy_Model, self).__init__()

        self.actor_std = nn.parameter.Parameter(
          torch.zeros(action_dim)
        )

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 64),
          nn.ReLU(),
          nn.Linear(64, action_dim)
        )
        
    def forward(self, states: Tensor, detach: bool = False) -> Tensor:
      mean  = self.nn_layer(states)
      std   = self.actor_std.exp()
      
      if detach:
        return (mean.detach(), std.detach())
      else:
        return (mean, std)
      
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
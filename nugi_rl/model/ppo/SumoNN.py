import torch
import torch.nn as nn
from torch import Tensor

from nugi_rl.model.components.SelfAttention import SelfAttention

class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, bins: int):
        super(Policy_Model, self).__init__()

        self.action_dim = action_dim
        self.bins = bins

        self.value  = nn.Linear(state_dim, state_dim)
        self.key    = nn.Linear(state_dim, state_dim)
        self.query  = nn.parameter.Parameter(
          torch.tensor(1, state_dim)
        )

        self.att  = SelfAttention()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 256),
          nn.SiLU(),
          nn.Linear(256, 128),
          nn.SiLU(),
          nn.Linear(128, action_dim * bins),
          nn.Sigmoid()
        )
        
    def forward(self, states: Tensor, detach: bool = False) -> tuple:
      value   = self.value(states)
      key     = self.key(states)

      context = self.att(value, key, self.query) 

      action = self.nn_layer(context)
      action = action.reshape(-1, self.action_dim, self.bins)

      if detach:
        return (action.detach(), )
      else:
        return (action, )

class Value_Model(nn.Module):
    def __init__(self, state_dim: int):
        super(Value_Model, self).__init__()

        self.value  = nn.Linear(state_dim, state_dim)
        self.key    = nn.Linear(state_dim, state_dim)
        self.query  = nn.parameter.Parameter(
          torch.tensor(1, state_dim)
        )

        self.att  = SelfAttention()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 64),
          nn.SiLU(),
          nn.Linear(64, 64),
          nn.SiLU(),
          nn.Linear(64, 1)
        )
        
    def forward(self, states: Tensor, detach: bool = False) -> Tensor:
      value   = self.value(states)
      key     = self.key(states)

      context = self.att(value, key, self.query)

      if detach:
        return self.nn_layer(context).detach()
      else:
        return self.nn_layer(context)
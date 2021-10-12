import torch
import torch.nn as nn
from helpers.pytorch_utils import set_device

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.actor_std = nn.parameter.Parameter(
          torch.zeros(action_dim)
        )

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, action_dim)
        )

        self.alpha = nn.parameter.Parameter(
          torch.ones(1)
        )

        self.beta = nn.parameter.Parameter(
          torch.ones(1)
        )
        
    def forward(self, states, detach = False):
      mean  = self.nn_layer(states)
      std   = self.actor_std.exp()
      
      if detach:
        return (mean.detach(), std.detach()), self.alpha.detach(), self.beta.detach()
      else:
        return (mean, std), self.alpha, self.beta
      
class Value_Model(nn.Module):
    def __init__(self, state_dim):
        super(Value_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, 1)
        )
        
    def forward(self, states, detach = False):
      if detach:
        return self.nn_layer(states).detach()
      else:
        return self.nn_layer(states)
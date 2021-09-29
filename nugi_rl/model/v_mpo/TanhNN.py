import torch
import torch.nn as nn
from helpers.pytorch_utils import set_device

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.temperature = nn.parameter.Parameter(
          torch.Tensor([1.0])
        )

        self.alpha_mean = nn.parameter.Parameter(
          torch.Tensor([1.0])
        )

        self.alpha_cov = nn.parameter.Parameter(
          torch.Tensor([1.0])
        )

        self.actor_std = nn.parameter.Parameter(
          torch.zeros(action_dim)
        )

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 512),
          nn.ReLU(),
          nn.Linear(512, 128),
          nn.ReLU(),
          nn.Linear(128, action_dim)
        )
        
    def forward(self, states, detach = False):
      mean    = self.nn_layer(states)
      std     = self.actor_std.exp()
      
      if detach:
        return (mean.detach(), std.detach()), self.temperature.detach(), (self.alpha_mean.detach(), self.alpha_cov.detach())
      else:
        return (mean, std), self.temperature, (self.alpha_mean, self.alpha_cov)
      
class Value_Model(nn.Module):
    def __init__(self, state_dim):
        super(Value_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 512),
          nn.ReLU(),
          nn.Linear(512, 128),
          nn.ReLU(),
          nn.Linear(128, 1)
        )
        
    def forward(self, states, detach = False):
      if detach:
        return self.nn_layer(states).detach()
      else:
        return self.nn_layer(states)
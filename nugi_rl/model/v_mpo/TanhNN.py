
import torch
import torch.nn as nn

from torch import Tensor

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

        self.actor_std  = nn.parameter.Parameter(
          torch.zeros(action_dim)
        )

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 64),
          nn.ReLU(),
          nn.Linear(64, action_dim)
        )
        
    def forward(self, states: Tensor) -> Tensor:
      mean  = self.nn_layer(states)
      std   = self.actor_std.exp()
      
      return (mean, std), self.temperature, (self.alpha_mean, self.alpha_cov)
      
class Value_Model(nn.Module):
    def __init__(self, state_dim):
        super(Value_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 64),
          nn.ReLU(),
          nn.Linear(64, 1)
        )
        
    def forward(self, states: Tensor) -> Tensor:
      return self.nn_layer(states)
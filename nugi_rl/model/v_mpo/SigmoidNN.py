
import torch
import torch.nn as nn

from torch import Tensor

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, bins):
        super(Policy_Model, self).__init__()

        self.action_dim = action_dim
        self.bins = bins

        self.temperature = nn.parameter.Parameter(
          torch.Tensor([1.0])
        )

        self.alpha_mean = nn.parameter.Parameter(
          torch.Tensor([1.0])
        )

        self.alpha_cov = nn.parameter.Parameter(
          torch.Tensor([1.0])
        )

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU(),
          nn.Linear(128, action_dim * bins),
          nn.Sigmoid()
        )
        
    def forward(self, states: Tensor) -> Tensor:
      action = self.nn_layer(states)
      action = action.reshape(-1, self.action_dim, self.bins)
      
      return action, self.temperature, (self.alpha_mean, self.alpha_cov)
      
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
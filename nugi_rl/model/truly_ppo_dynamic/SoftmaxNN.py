import torch
import torch.nn as nn
from helpers.pytorch_utils import set_device

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 32),
          nn.ReLU(),
          nn.Linear(32, 32),
          nn.ReLU(),
          nn.Linear(32, action_dim),
          nn.Softmax(-1)
        )

        self.alpha = nn.parameter.Parameter(
          torch.ones(action_dim)
        )

        self.beta = nn.parameter.Parameter(
          torch.ones(action_dim)
        )
        
    def forward(self, states, detach = False):
      if detach:
        return self.nn_layer(states).detach(), self.alpha.detach(), self.beta.detach()
      else:
        return self.nn_layer(states), self.alpha, self.beta

class Value_Model(nn.Module):
    def __init__(self, state_dim):
        super(Value_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 32),
          nn.ReLU(),
          nn.Linear(32, 32),
          nn.ReLU(),
          nn.Linear(32, 1)
        )
        
    def forward(self, states, detach = False):
      if detach:
        return self.nn_layer(states).detach()
      else:
        return self.nn_layer(states)
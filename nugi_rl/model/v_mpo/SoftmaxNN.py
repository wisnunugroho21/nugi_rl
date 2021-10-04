import torch
import torch.nn as nn
from helpers.pytorch_utils import set_device

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Policy_Model, self).__init__()

        self.temperature = nn.parameter.Parameter(
          torch.Tensor([1.0])
        )

        self.alpha = nn.parameter.Parameter(
          torch.Tensor([1.0])
        )

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU()
        )

        self.actor_layer = nn.Sequential(
          nn.Linear(64, action_dim),
          nn.Softmax(-1)
        )
        
    def forward(self, states, detach = False):
      x = self.nn_layer(states)

      if detach:
        return self.actor_layer(x).detach(), self.temperature.detach(), self.alpha.detach()
      else:
        return self.actor_layer(x), self.temperature, self.alpha

class Value_Model(nn.Module):
    def __init__(self, state_dim, use_gpu = True):
        super(Value_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU(),
          nn.Linear(64, 1)
        )
        
    def forward(self, states, detach = False):
      if detach:
        return self.nn_layer(states).detach()
      else:
        return self.nn_layer(states)
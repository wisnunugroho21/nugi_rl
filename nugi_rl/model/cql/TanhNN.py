import torch
import torch.nn as nn
from helpers.pytorch_utils import set_device

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, action_dim),
          nn.Tanh()
        )
        
    def forward(self, states, detach = False):
      action = self.nn_layer(states)

      if detach:
        return action.detach()
      else:
        return action
      
class Q_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Q_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim + action_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, 1)
        )
        
    def forward(self, states, actions, detach = False):
      x   = torch.cat((states, actions), -1)

      if detach:
        return self.nn_layer(x).detach()
      else:
        return self.nn_layer(x)

class Value_Model(nn.Module):
    def __init__(self, state_dim, use_gpu = True):
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
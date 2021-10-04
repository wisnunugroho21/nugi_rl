import torch
import torch.nn as nn
from helpers.pytorch_utils import set_device

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.action_dim = action_dim

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU()
        )

        self.mean_layer = nn.Sequential(
          nn.Linear(64, action_dim),
        )

        self.std_layer = nn.Sequential(
          nn.Linear(64, action_dim)
        )
        
    def forward(self, states, detach = False):
      x     = self.nn_layer(states)
      mean  = self.mean_layer(x[:, :64])
      std   = self.std_layer(x[:, 64:]).exp()
      
      if detach:
        return (mean.detach(), std.detach())
      else:
        return (mean, std)
      
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
        
    def forward(self, states, detach = False):
      if detach:
        return self.nn_layer(states).detach()
      else:
        return self.nn_layer(states)
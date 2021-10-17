import torch
import torch.nn as nn
from helpers.pytorch_utils import set_device

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU(),
        )

        self.actor_layer = nn.Sequential(
          nn.Linear(64, action_dim),
        )

        self.actor_std_layer = nn.Sequential(
          nn.Linear(64, action_dim)
        )
        
    def forward(self, states, detach = False):
      x     = self.nn_layer(states)
      mean  = self.actor_layer(x[:, :64])
      std   = self.actor_std_layer(x[:, 64:]).exp()

      if detach:
        return (mean.detach(), std.detach())
      else:
        return (mean, std)
      
class Q_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim + action_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 64),
          nn.ReLU(),
          nn.Linear(64, 1)
        )
        
    def forward(self, states, actions, detach = False):
      x   = torch.cat((states, actions), -1)

      if detach:
        return self.nn_layer(x).detach()
      else:
        return self.nn_layer(x)
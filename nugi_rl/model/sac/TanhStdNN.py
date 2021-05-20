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
        ).float().to(set_device(use_gpu))

        self.actor_layer = nn.Sequential(
          nn.Linear(32, action_dim),
          nn.Tanh()
        ).float().to(set_device(use_gpu))

        self.actor_std_layer = nn.Sequential(
          nn.Linear(32, action_dim),
          nn.Sigmoid()
        ).float().to(set_device(use_gpu))
        
    def forward(self, states, detach = False):
      x = self.nn_layer(states)

      if detach:
        return (self.actor_layer(x[:, :32]).detach(), self.actor_std_layer(x[:, 32:64]).detach())
      else:
        return (self.actor_layer(x[:, :32]), self.actor_std_layer(x[:, 32:64]))
      
class Q_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Q_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim + action_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 32),
          nn.ReLU(),
          nn.Linear(32, 1)
        ).float().to(set_device(use_gpu))
        
    def forward(self, states, actions, detach = False):
      x   = torch.cat((states, actions), -1)

      if detach:
        return self.nn_layer(x).detach()
      else:
        return self.nn_layer(x)
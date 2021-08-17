import torch
import torch.nn as nn
from helpers.pytorch_utils import set_device

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 1024),
          nn.ReLU(),
          nn.Linear(1024, 768),
          nn.ReLU(),
        )

        self.actor_mean_layer = nn.Sequential(
          nn.Linear(256, action_dim)
        )

        self.actor_std_layer = nn.Sequential(
          nn.Linear(256, action_dim),
          nn.Sigmoid()
        )

        self.critic_layer = nn.Sequential(
          nn.Linear(256, 1)
        )
        
    def forward(self, states, detach = False):
      x = self.nn_layer(states)

      mean    = self.actor_mean_layer(x[:, :256])
      std     = self.actor_std_layer(x[:, 256:512])
      critic  = self.critic_layer(x[:, 512:768])
      
      if detach:
        return (mean.detach(), std.detach()), critic.detach()
      else:
        return (mean, std), critic
      
class Value_Model(nn.Module):
    def __init__(self, state_dim):
        super(Value_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 1024),
          nn.ReLU(),
          nn.Linear(1024, 256),
          nn.ReLU(),
          nn.Linear(256, 1)
        )
        
    def forward(self, states, detach = False):
      if detach:
        return self.nn_layer(states).detach()
      else:
        return self.nn_layer(states)
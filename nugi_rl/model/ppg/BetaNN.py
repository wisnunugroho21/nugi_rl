import torch
import torch.nn as nn
from helpers.pytorch_utils import set_device

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 96),
          nn.ReLU(),
        )

        self.actor_alpha_layer = nn.Sequential(
          nn.Linear(32, action_dim),
          nn.Softplus()
        )

        self.actor_beta_layer = nn.Sequential(
          nn.Linear(32, action_dim),
          nn.Softplus()
        )

        self.critic_layer = nn.Sequential(
          nn.Linear(32, 1)
        )
        
    def forward(self, states, detach = False):
      x       = self.nn_layer(states)
      
      alpha   = self.actor_alpha_layer(x[:, :32])
      beta    = self.actor_beta_layer(x[:, 32:64])
      critic  = self.critic_layer(x[:, 64:96])

      if detach:
        return (alpha.detach(), beta.detach()), critic.detach()
      else:
        return (alpha, beta), critic
      
class Value_Model(nn.Module):
    def __init__(self, state_dim, use_gpu = True):
        super(Value_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 32),
          nn.ReLU(),
          nn.Linear(32, 1)
        )
        
    def forward(self, states, detach = False):
      if detach:
        return self.nn_layer(states).detach()
      else:
        return self.nn_layer(states)
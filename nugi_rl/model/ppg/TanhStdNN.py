import torch
import torch.nn as nn

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 192),
          nn.ReLU(),
        )

        self.actor_mean_layer = nn.Sequential(
          nn.Linear(64, action_dim)
        )

        self.actor_std_layer = nn.Sequential(
          nn.Linear(64, action_dim),
          nn.Sigmoid()
        )

        self.critic_layer = nn.Sequential(
          nn.Linear(64, 1)
        )
        
    def forward(self, states, detach = False):
      x = self.nn_layer(states)

      mean    = self.actor_mean_layer(x[:, :64])
      std     = self.actor_std_layer(x[:, 64:128])
      critic  = self.critic_layer(x[:, 128:])
      
      if detach:
        return (mean.detach(), std.detach()), critic.detach()
      else:
        return (mean, std), critic
      
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
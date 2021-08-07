import torch
import torch.nn as nn
from helpers.pytorch_utils import set_device

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 128),
          nn.ReLU(),
        )

        self.actor_layer = nn.Sequential(
          nn.Linear(64, action_dim),
          nn.Tanh()
        )

        self.critic_layer = nn.Sequential(
          nn.Linear(64, 1)
        )

        self.std = torch.FloatTensor([1.0])
        
    def forward(self, states, detach = False):
      x = self.nn_layer(states)

      if detach:
        return (self.actor_layer(x[:, :64]).detach(), self.std.detach()), self.critic_layer(x[:, 64:128]).detach()
      else:
        return (self.actor_layer(x[:, :64]), self.std), self.critic_layer(x[:, 64:128])
      
class Value_Model(nn.Module):
    def __init__(self, state_dim):
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
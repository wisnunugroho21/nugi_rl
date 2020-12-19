import torch
import torch.nn as nn
from utils.pytorch_utils import set_device

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
              ).float().to(set_device(use_gpu))

        self.actor_layer = nn.Sequential(
                nn.Linear(128, action_dim),
                nn.Tanh()
              ).float().to(set_device(use_gpu))

        self.critic_layer = nn.Sequential(
                nn.Linear(128, 1)
              ).float().to(set_device(use_gpu))
        
    def forward(self, states):
        x = self.nn_layer(states)
        return self.actor_layer(x), self.critic_layer(x)

class Value_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Value_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
                 nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
              ).float().to(set_device(use_gpu))
        
    def forward(self, states):
        return self.nn_layer(states)
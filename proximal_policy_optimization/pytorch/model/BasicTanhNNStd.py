import torch
import torch.nn as nn
from utils.pytorch_utils import set_device

class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Actor_Model, self).__init__()   

        self.mean_layer = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh()
              ).float().to(set_device(use_gpu))

        self.std_layer = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
              ).float().to(set_device(use_gpu))
        
    def forward(self, states):
        return self.mean_layer(states), self.std_layer(states)

class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Critic_Model, self).__init__() 

        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
              ).float().to(set_device(use_gpu))
        
    def forward(self, states):        
        return self.nn_layer(states)
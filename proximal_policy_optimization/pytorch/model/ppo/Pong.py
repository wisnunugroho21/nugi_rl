import torch
import torch.nn as nn
from utils.pytorch_utils import set_device

class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Actor_Model, self).__init__()  

        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 640),
                nn.ReLU(),
                nn.Linear(640, 640),
                nn.ReLU(),
                nn.Linear(640, action_dim),
                nn.Softmax(-1)
              ).float().to(set_device(use_gpu))
        
    def forward(self, states):
        return self.nn_layer(states)

class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Critic_Model, self).__init__() 

        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 640),
                nn.ReLU(),
                nn.Linear(640, 640),
                nn.ReLU(),
                nn.Linear(640, 1)
              ).float().to(set_device(use_gpu))
        
    def forward(self, states):
        return self.nn_layer(states)
import torch
import torch.nn as nn
from utils.pytorch_utils import set_device

class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Actor_Model, self).__init__()  

        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 10)        
              ).float().to(set_device(use_gpu))

        self.embedded_layer = nn.Embedding(1028, 10).float().to(set_device(use_gpu))
        self.softmax = nn.Softmax(-1).float().to(set_device(use_gpu))
        
    def forward(self, states, available_actions):
        x1 = self.nn_layer(states)
        x2 = self.embedded_layer(available_actions).squeeze(0)
        x = torch.mm(x1, x2.transpose(0, 1))

        return self.softmax(x)

class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Critic_Model, self).__init__() 

        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
              ).float().to(set_device(use_gpu))
        
    def forward(self, states):
        return self.nn_layer(states)
import torch
import torch.nn as nn
from utils.pytorch_utils import set_device
      
class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Actor_Model, self).__init__()   

      self.use_gpu = use_gpu

      self.nn_layer = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        nn.Flatten(),        
        nn.Linear(800, 400),
        nn.ReLU(),
        nn.Linear(400, action_dim),
        nn.Softmax(-1)     
      ).float().to(set_device(use_gpu))      
        
    def forward(self, states):
        return self.nn_layer(states)

class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Critic_Model, self).__init__()

      self.use_gpu = use_gpu

      self.nn_layer = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        nn.Flatten(),        
        nn.Linear(800, 400),
        nn.ReLU(),
        nn.Linear(400, 1)       
      ).float().to(set_device(use_gpu))
        
    def forward(self, states):
        return self.nn_layer(states)
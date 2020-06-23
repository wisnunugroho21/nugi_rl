import torch
import torch.nn as nn
from utils.pytorch_utils import set_device

class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Actor_Model, self).__init__()   

      self.use_gpu = use_gpu
      self.memory_layer = nn.LSTM(state_dim, 640).float().to(set_device(use_gpu))
      
      self.nn_layer = nn.Sequential(
        nn.Linear(640, 640),
        nn.ReLU(),
        nn.Linear(640, action_dim),
        nn.Softmax(-1)
      ).float().to(set_device(use_gpu))
        
    def forward(self, states):
      r_in              = states.transpose(0, 1)
      r_out, (h_n, c_n) = self.memory_layer(r_in)

      last_in = r_out[-1]
      return self.nn_layer(last_in)

class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Critic_Model, self).__init__()

      self.use_gpu = use_gpu
      self.memory_layer = nn.LSTM(state_dim, 640).float().to(set_device(use_gpu))
      
      self.nn_layer = nn.Sequential(
        nn.Linear(640, 640),
        nn.ReLU(),
        nn.Linear(640, 1)
      ).float().to(set_device(use_gpu))
        
    def forward(self, states):
      r_in              = states.transpose(0, 1)
      r_out, (h_n, c_n) = self.memory_layer(r_in)

      last_in = r_out[-1]
      return self.nn_layer(last_in)
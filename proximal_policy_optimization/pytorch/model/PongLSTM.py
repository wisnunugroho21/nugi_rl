import torch
import torch.nn as nn
from utils.pytorch_utils import set_device

class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Actor_Model, self).__init__()   

      self.use_gpu = use_gpu

      self.input_layer  = nn.Sequential(
        nn.Linear(state_dim, 640),
        nn.ReLU()
      ).float().to(set_device(use_gpu))

      self.memory_layer = nn.LSTM(640, 320).float().to(set_device(use_gpu))
      
      self.nn_layer = nn.Sequential(
        nn.Linear(320, 320),
        nn.ReLU(),
        nn.Linear(320, action_dim),
        nn.Softmax(-1)
      ).float().to(set_device(use_gpu))
        
    def forward(self, states):  
      batch_size, timesteps, F = states.size()

      c_in  = states.transpose(0, 1).reshape(timesteps * batch_size, F)
      c_out = self.input_layer(c_in)

      r_in                = c_out.reshape(timesteps, batch_size, c_out.shape[-1])
      r_out, (h_n, c_n)   = self.memory_layer(r_in)

      last_in = r_out[-1]
      return self.nn_layer(last_in)

class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Critic_Model, self).__init__()

      self.use_gpu = use_gpu

      self.input_layer  = nn.Sequential(
        nn.Linear(state_dim, 640),
        nn.ReLU()
      ).float().to(set_device(use_gpu))

      self.memory_layer = nn.LSTM(640, 320).float().to(set_device(use_gpu))
      
      self.nn_layer = nn.Sequential(
        nn.Linear(320, 320),
        nn.ReLU(),
        nn.Linear(320, 1)
      ).float().to(set_device(use_gpu))
        
    def forward(self, states):
      batch_size, timesteps, F = states.size()

      c_in  = states.transpose(0, 1).reshape(timesteps * batch_size, F)
      c_out = self.input_layer(c_in)

      r_in              = c_out.reshape(timesteps, batch_size, c_out.shape[-1])
      r_out, (h_n, c_n) = self.memory_layer(r_in)

      last_in = r_out[-1]
      return self.nn_layer(last_in)
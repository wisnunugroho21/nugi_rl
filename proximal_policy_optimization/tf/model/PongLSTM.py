import torch
import torch.nn as nn
from utils.pytorch_utils import set_device

class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Actor_Model, self).__init__()   

      self.use_gpu = use_gpu

      self.input_layer = nn.Sequential(
        nn.Linear(25600, 1600),
        nn.ReLU(),  
        nn.Linear(1600, 200),
        nn.ReLU()   
      ).float().to(set_device(use_gpu))

      self.memory_layer = nn.LSTM(200, 200).float().to(set_device(use_gpu))

      self.nn_layer = nn.Sequential(
        nn.Linear(200, 25),
        nn.ReLU(),
        nn.Linear(25, action_dim),
        nn.Softmax(-1)
      ).float().to(set_device(use_gpu))        
        
    def forward(self, states):  
      batch_size, timesteps, S = states.size()

      c_in  = states.view(batch_size * timesteps, S)
      c_out = self.input_layer(c_in)

      r_in              = c_out.view(-1, batch_size, c_out.shape[-1])
      hidden            = torch.full((1, batch_size, c_out.shape[-1]), 0.01).float().to(set_device(self.use_gpu))
      r_out, (h_n, h_c) = self.memory_layer(r_in, (hidden, hidden))

      last_in = r_out[-1]
      return self.nn_layer(last_in) 

class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Critic_Model, self).__init__()

      self.use_gpu = use_gpu

      self.input_layer = nn.Sequential(
        nn.Linear(25600, 1600),
        nn.ReLU(),  
        nn.Linear(1600, 200),
        nn.ReLU()   
      ).float().to(set_device(use_gpu)) 

      self.memory_layer = nn.LSTM(200, 200).float().to(set_device(use_gpu))

      self.nn_layer = nn.Sequential(
        nn.Linear(200, 25),
        nn.ReLU(),
        nn.Linear(25, 1)
      ).float().to(set_device(use_gpu))
        
    def forward(self, states):
      batch_size, timesteps, S = states.size()

      c_in  = states.view(batch_size * timesteps, S)
      c_out = self.input_layer(c_in)

      r_in              = c_out.view(-1, batch_size, c_out.shape[-1])
      hidden            = torch.full((1, batch_size, c_out.shape[-1]), 0.01).float().to(set_device(self.use_gpu))
      r_out, (h_n, h_c) = self.memory_layer(r_in, (hidden, hidden))

      last_in = r_out[-1]
      return self.nn_layer(last_in)
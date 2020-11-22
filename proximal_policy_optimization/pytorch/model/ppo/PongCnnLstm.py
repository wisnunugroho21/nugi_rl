import torch
import torch.nn as nn
from utils.pytorch_utils import set_device
from model.SeperableConv2d import DepthwiseSeparableConv2d, SeparableConv2d

class CNN_Model(nn.Module):
    def __init__(self):
      super(CNN_Model, self).__init__()   

      self.conv1 = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        SeparableConv2d(8, 8, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU()
      )

      self.conv2 = nn.Sequential(
        SeparableConv2d(8, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        SeparableConv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.MaxPool2d(2, 2)          
      )      

      self.conv3 = nn.Sequential(
        SeparableConv2d(8, 16, kernel_size = 4, stride = 2, padding = 1)
      )

      self.out_layer = nn.Sequential(
        nn.ReLU(),
        nn.Flatten(),        
        nn.Linear(6400, 640),
        nn.ReLU()
      )
        
    def forward(self, states):
      x1 = self.conv1(states)
      x2 = self.conv2(x1)
      x3 = self.conv3(x1)
      return self.out_layer(x2 + x3)

class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Actor_Model, self).__init__()   

      self.use_gpu = use_gpu

      self.input_layer  = CNN_Model().float().to(set_device(use_gpu))
      self.memory_layer = nn.LSTM(640, 320).float().to(set_device(use_gpu))
      self.nn_layer     = nn.Sequential(
        nn.Linear(320, 100),
        nn.ReLU(),
        nn.Linear(100, action_dim),
        nn.Softmax(-1)
      ).float().to(set_device(use_gpu))        
        
    def forward(self, states):
      batch_size, timesteps, C, H, W = states.size()

      c_in  = states.transpose(0, 1).reshape(timesteps * batch_size, C, H, W)
      c_out = self.input_layer(c_in)

      r_in                = c_out.reshape(timesteps, batch_size, c_out.shape[-1])
      r_out, (h_n, c_n)   = self.memory_layer(r_in)

      last_in = r_out[-1]
      return self.nn_layer(last_in)

class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Critic_Model, self).__init__()

      self.use_gpu = use_gpu

      self.input_layer  = CNN_Model().float().to(set_device(use_gpu))
      self.memory_layer = nn.LSTM(640, 320).float().to(set_device(use_gpu))
      self.nn_layer     = nn.Sequential(
        nn.Linear(320, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
      ).float().to(set_device(use_gpu))
        
    def forward(self, states):  
      batch_size, timesteps, C, H, W = states.size()

      c_in  = states.transpose(0, 1).reshape(timesteps * batch_size, C, H, W)
      c_out = self.input_layer(c_in)

      r_in                = c_out.reshape(timesteps, batch_size, c_out.shape[-1])
      r_out, (h_n, c_n)   = self.memory_layer(r_in)

      last_in = r_out[-1]
      return self.nn_layer(last_in)
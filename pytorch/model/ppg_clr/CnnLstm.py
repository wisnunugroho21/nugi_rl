import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pytorch_utils import set_device
from model.components.SeperableConv2d import DepthwiseSeparableConv2d

class CnnModel(nn.Module):
    def __init__(self, use_gpu = True):
      super(CnnModel, self).__init__()   

      self.conv1 = nn.Sequential(
        DepthwiseSeparableConv2d(3, 16, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 32, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      ).to(set_device(use_gpu))

      self.conv2 = nn.Sequential(
        DepthwiseSeparableConv2d(32, 64, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      ).to(set_device(use_gpu))

      self.conv3 = nn.Sequential(
        DepthwiseSeparableConv2d(32, 128, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
      ).to(set_device(use_gpu))

      self.conv_out = nn.Sequential(
        DepthwiseSeparableConv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      ).to(set_device(use_gpu))
        
    def forward(self, states, detach = False):
      timestep = 1
      if len(states.shape) == 5:
        batch, timestep, C, H, W = states.shape
        states = states.reshape(batch * timestep, C, H, W)

      i1  = self.conv1(states)
      i2  = self.conv2(i1)
      i3  = self.conv3(i1)
      i23 = i2 + i3
      out  = self.conv_out(i23)

      if timestep > 1:
        if detach:
          return out.mean([2, 3]).reshape(batch, timestep, -1).detach()
        else:
          return out.mean([2, 3]).reshape(batch, timestep, -1)
      else:
        if detach:
          return out.mean([2, 3]).detach()
        else:
          return out.mean([2, 3])

class ProjectionModel(nn.Module):
    def __init__(self, use_gpu = True):
      super(ProjectionModel, self).__init__()

      self.nn_layer   = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256)
      ).to(set_device(use_gpu))

    def forward(self, states, detach = False):
      if detach:
        return self.nn_layer(states).detach()
      else:
        return self.nn_layer(states)

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Policy_Model, self).__init__()

      self.memory_layer   = nn.LSTM(256, 256).float().to(set_device(use_gpu))
      self.nn_layer       = nn.Sequential( nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU() ).float().to(set_device(use_gpu))

      self.critic_layer   = nn.Sequential( nn.Linear(64, 1) ).float().to(set_device(use_gpu))
      self.actor_layer    = nn.Sequential( nn.Linear(64, action_dim), nn.Softmax(-1) ).float().to(set_device(use_gpu))
        
    def forward(self, datas, detach = False):
      batch_size, timesteps, S  = datas.shape

      m         = datas.transpose(0, 1).reshape(timesteps, batch_size, S)
      m, (h, c) = self.memory_layer(m)

      x   = m[-1]
      x   = self.nn_layer(x)

      if detach:
        return self.actor_layer(x).detach(), self.critic_layer(x).detach()
      else:
        return self.actor_layer(x), self.critic_layer(x)
      
class Value_Model(nn.Module):
    def __init__(self, state_dim, use_gpu = True):
      super(Value_Model, self).__init__()

      self.memory_layer   = nn.LSTM(256, 256).float().to(set_device(use_gpu))
      self.nn_layer       = nn.Sequential( nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU() ).float().to(set_device(use_gpu))

      self.critic_layer   = nn.Sequential( nn.Linear(64, 1) ).float().to(set_device(use_gpu))
        
    def forward(self, datas, detach = False):
      batch_size, timesteps, S  = datas.shape
      
      m         = datas.transpose(0, 1).reshape(timesteps, batch_size, S)
      m, (h, c) = self.memory_layer(m)

      x   = m[-1]
      x   = self.nn_layer(x)

      if detach:
        return self.critic_layer(x).detach()
      else:
        return self.critic_layer(x)
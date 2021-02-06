import torch
import torch.nn as nn
from utils.pytorch_utils import set_device
from model.components.SeperableConv2d import DepthwiseSeparableConv2d

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Policy_Model, self).__init__()

      self.conv = nn.Sequential(
        DepthwiseSeparableConv2d(3, 16, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 32, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(32, 64, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.nn_layer = nn.Sequential(
        nn.Linear(256, 64),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.actor_layer = nn.Sequential(
        nn.Linear(64, action_dim),
        nn.Tanh()
      ).float().to(set_device(use_gpu))

      self.critic_layer = nn.Sequential(
        nn.Linear(64, 1)
      ).float().to(set_device(use_gpu))
        
    def forward(self, x):
      x = x.transpose(2, 3).transpose(1, 2)

      x = self.conv(x)
      x = x.mean([2, 3])
      x = self.nn_layer(x)

      return self.actor_layer(x), self.critic_layer(x)
      
class Value_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Value_Model, self).__init__()   

      self.conv = nn.Sequential(
        DepthwiseSeparableConv2d(3, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 16, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),

        DepthwiseSeparableConv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(32, 32, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),

        DepthwiseSeparableConv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(64, 64, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),

        DepthwiseSeparableConv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 128, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.nn_layer = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.critic_layer = nn.Sequential(
        nn.Linear(64, 1)
      ).float().to(set_device(use_gpu))
        
    def forward(self, x):
      x = x.transpose(2, 3).transpose(1, 2)

      x = self.conv(x)
      x = x.mean([2, 3])
      x = self.nn_layer(x)

      return self.critic_layer(x)
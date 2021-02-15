import torch
import torch.nn as nn
from utils.pytorch_utils import set_device
from model.components.SeperableConv2d import DepthwiseSeparableConv2d

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Policy_Model, self).__init__()

      self.bn1 = nn.BatchNorm2d(3).float().to(set_device(use_gpu))

      self.conv1 = nn.Sequential(
        DepthwiseSeparableConv2d(3, 16, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 64, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),        
      ).float().to(set_device(use_gpu))

      self.conv2 = nn.Sequential(
        DepthwiseSeparableConv2d(3, 64, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.conv3 = nn.Sequential(
        DepthwiseSeparableConv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.conv4 = nn.Sequential(
        DepthwiseSeparableConv2d(64, 256, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.state_extractor = nn.Sequential(
        nn.Linear(1, 256),
        nn.ReLU()
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
        
    def forward(self, images, states):
      x = images
      x = x.transpose(2, 3).transpose(1, 2)
      x = self.bn1(x)

      x1  = self.conv1(x)
      x2  = self.conv2(x)
      x12 = x1 + x2

      x3  = self.conv3(x12)
      x4  = self.conv4(x12)
      x34 = x3 + x4

      x5 = x34.mean([2, 3])

      x6 = self.state_extractor(states)
      x56 = x5 + x6
      x7   = self.nn_layer(x56)

      return self.actor_layer(x7), self.critic_layer(x7)
      
class Value_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Value_Model, self).__init__()

      self.bn1 = nn.BatchNorm2d(3).float().to(set_device(use_gpu))   

      self.conv1 = nn.Sequential(
        DepthwiseSeparableConv2d(3, 16, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 64, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),        
      ).float().to(set_device(use_gpu))

      self.conv2 = nn.Sequential(
        DepthwiseSeparableConv2d(3, 64, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.conv3 = nn.Sequential(
        DepthwiseSeparableConv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.conv4 = nn.Sequential(
        DepthwiseSeparableConv2d(64, 256, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.state_extractor = nn.Sequential(
        nn.Linear(1, 256),
        nn.ReLU()
      ).float().to(set_device(use_gpu))

      self.nn_layer = nn.Sequential(
        nn.Linear(256, 64),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.critic_layer = nn.Sequential(
        nn.Linear(64, 1)
      ).float().to(set_device(use_gpu))
        
    def forward(self, images, states):
      x = images
      x = x.transpose(2, 3).transpose(1, 2)
      x = self.bn1(x)

      x1  = self.conv1(x)
      x2  = self.conv2(x)
      x12 = x1 + x2

      x3  = self.conv3(x12)
      x4  = self.conv4(x12)
      x34 = x3 + x4

      x5 = x34.mean([2, 3])

      x6 = self.state_extractor(states)
      x56 = x5 + x6
      x7   = self.nn_layer(x56)

      return self.critic_layer(x7)
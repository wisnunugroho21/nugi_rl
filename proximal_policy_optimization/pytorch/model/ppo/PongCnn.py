import torch
import torch.nn as nn
from utils.pytorch_utils import set_device
from model.SeperableConv2d import DepthwiseSeparableConv2d, SeparableConv2d

class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Actor_Model, self).__init__()   

      self.use_gpu = use_gpu      

      self.conv1 = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        SeparableConv2d(8, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU()
      ).float().to(set_device(use_gpu))

      self.conv2 = nn.Sequential(
        SeparableConv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        SeparableConv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.MaxPool2d(2, 2)          
      ).float().to(set_device(use_gpu))        

      self.conv3 = nn.Sequential(
        SeparableConv2d(16, 16, kernel_size = 4, stride = 2, padding = 1)
      ).float().to(set_device(use_gpu))

      self.out_layer = nn.Sequential(
        nn.ReLU(),
        nn.Flatten(),        
        nn.Linear(6400, 640),
        nn.ReLU(),
        nn.Linear(640, action_dim),
        nn.Softmax(-1)     
      ).float().to(set_device(use_gpu))      
        
    def forward(self, states):
      x1 = self.conv1(states)
      x2 = self.conv2(x1)
      x3 = self.conv3(x1)
      x4 = x2 + x3
      return self.out_layer(x4)

class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Critic_Model, self).__init__()

      self.use_gpu = use_gpu

      self.conv1 = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        SeparableConv2d(8, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU()
      ).float().to(set_device(use_gpu))

      self.conv2 = nn.Sequential(
        SeparableConv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        SeparableConv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.MaxPool2d(2, 2)          
      ).float().to(set_device(use_gpu))        

      self.conv3 = nn.Sequential(
        SeparableConv2d(16, 16, kernel_size = 4, stride = 2, padding = 1)
      ).float().to(set_device(use_gpu))

      self.out_layer = nn.Sequential(
        nn.ReLU(),
        nn.Flatten(),        
        nn.Linear(6400, 640),
        nn.ReLU(),
        nn.Linear(640, 1)       
      ).float().to(set_device(use_gpu))
        
    def forward(self, states):
      x1 = self.conv1(states)
      x2 = self.conv2(x1)
      x3 = self.conv3(x1)
      x4 = x2 + x3
      return self.out_layer(x4)
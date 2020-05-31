import torch
import torch.nn as nn
from utils.pytorch_utils import set_device

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, stride = 1, padding = 0, dilation = 1, depth_multiplier = 1):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        self.nn_layer = nn.Sequential(
            nn.Conv2d(nin, nin * depth_multiplier, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = nin),
            nn.Conv2d(nin * depth_multiplier, nout, kernel_size = 1)
          )

    def forward(self, x):
        return self.nn_layer(x)

class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Actor_Model, self).__init__()   

      self.use_gpu = use_gpu

      self.conv_layer1 = nn.Sequential(
        DepthwiseSeparableConv2d(3, 8, kernel_size = 4, stride = 2, padding = 1)        
      ).float().to(set_device(use_gpu))

      self.conv_layer2 = nn.Sequential(
        nn.ReLU(),
        DepthwiseSeparableConv2d(8, 8, kernel_size = 3, stride = 1, padding = 1)
      ).float().to(set_device(use_gpu))

      self.conv_layer3 = nn.Sequential(        
        DepthwiseSeparableConv2d(8, 16, kernel_size = 4, stride = 2, padding = 1)
      ).float().to(set_device(use_gpu))

      self.conv_layer3 = nn.Sequential(      
        nn.ReLU(),  
        DepthwiseSeparableConv2d(16, 16, kernel_size = 4, stride = 2, padding = 1)
      ).float().to(set_device(use_gpu))

      self.out_layer = nn.Sequential(
        nn.Flatten(),        
        nn.Linear(7056, 640),
        nn.ReLU(),
        nn.Linear(640, action_dim),
        nn.Softmax(-1)     
      ).float().to(set_device(use_gpu))      
        
    def forward(self, states):
      x1 = self.conv_layer1(states)
      x2 = self.conv_layer2(x1)
      x3 = x1 + x2
      x4 = self.conv_layer3(x3)
      x5 = self.conv_layer4(x4)
      x6 = x4 + x5
      return self.out_layer(x6)

class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Critic_Model, self).__init__()

      self.use_gpu = use_gpu

      self.conv_layer1 = nn.Sequential(
        DepthwiseSeparableConv2d(3, 8, kernel_size = 4, stride = 2, padding = 1)        
      ).float().to(set_device(use_gpu))

      self.conv_layer2 = nn.Sequential(
        nn.ReLU(),
        DepthwiseSeparableConv2d(8, 8, kernel_size = 3, stride = 1, padding = 1)
      ).float().to(set_device(use_gpu))

      self.conv_layer3 = nn.Sequential(        
        DepthwiseSeparableConv2d(8, 16, kernel_size = 4, stride = 2, padding = 1)
      ).float().to(set_device(use_gpu))

      self.conv_layer3 = nn.Sequential(      
        nn.ReLU(),  
        DepthwiseSeparableConv2d(16, 16, kernel_size = 3, stride = 1, padding = 1)
      ).float().to(set_device(use_gpu))

      self.out_layer = nn.Sequential(
        nn.Flatten(),        
        nn.Linear(7056, 640),
        nn.ReLU(),
        nn.Linear(640, 1)       
      ).float().to(set_device(use_gpu))
        
    def forward(self, states):
      x1 = self.conv_layer1(states)
      x2 = self.conv_layer2(x1)
      x3 = x1 + x2
      x4 = self.conv_layer3(x3)
      x5 = self.conv_layer4(x4)
      x6 = x4 + x5
      return self.out_layer(x6)
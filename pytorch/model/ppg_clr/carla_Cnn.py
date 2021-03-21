import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pytorch_utils import set_device
from model.components.SeperableConv2d import DepthwiseSeparableConv2d
from model.components.ASPP import AtrousSpatialPyramidConv2d

class CnnModel(nn.Module):
    def __init__(self):
      super(CnnModel, self).__init__()   

      self.bn1 = nn.BatchNorm2d(32)
      self.bn2 = nn.BatchNorm2d(64)

      self.conv1 = nn.Sequential(
        AtrousSpatialPyramidConv2d(3, 8),
        nn.ReLU(),
        DepthwiseSeparableConv2d(8, 16, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      )

      self.conv2 = nn.Sequential(
        DepthwiseSeparableConv2d(16, 16, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 32, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
      )

      self.conv3 = nn.Sequential(
        DepthwiseSeparableConv2d(16, 32, kernel_size = 8, stride = 4, padding = 2, bias = False),
        nn.ReLU(),
      )

      self.conv4 = nn.Sequential(
        DepthwiseSeparableConv2d(32, 32, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(32, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
      )

      self.conv5 = nn.Sequential(
        DepthwiseSeparableConv2d(32, 64, kernel_size = 8, stride = 4, padding = 2, bias = False),
        nn.ReLU(),
      )

      self.conv_out = nn.Sequential(
        DepthwiseSeparableConv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      )
        
    def forward(self, state, detach = False):
      i1  = self.conv1(state)
      i2  = self.conv2(i1)
      i3  = self.conv3(i1)
      i23 = self.bn1(i2 + i3)
      i4  = self.conv4(i23)
      i5  = self.conv5(i23)
      i45 = self.bn2(i4 + i5)
      out = self.conv_out(i45)
      out = out.mean([-1, -2])

      if detach:
        return out.detach()
      else:
        return out

class ProjectionModel(nn.Module):
    def __init__(self):
      super(ProjectionModel, self).__init__()

      self.nn_layer   = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
      )

    def forward(self, state, detach = False):      
      if detach:
        return self.nn_layer(state).detach()
      else:
        return self.nn_layer(state)

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Policy_Model, self).__init__()

      self.std                  = torch.FloatTensor([1.0, 0.5, 0.5]).to(set_device(use_gpu))

      self.nn_layer             = nn.Sequential( nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU() )

      self.critic_layer         = nn.Sequential( nn.Linear(64, 1) )
      self.actor_tanh_layer     = nn.Sequential( nn.Linear(64, 1), nn.Tanh() )
      self.actor_sigmoid_layer  = nn.Sequential( nn.Linear(64, 2), nn.Sigmoid() )            
        
    def forward(self, state, detach = False):
      x   = self.nn_layer(state)

      action_tanh     = self.actor_tanh_layer(x)
      action_sigmoid  = self.actor_sigmoid_layer(x)
      action          = torch.cat((action_tanh, action_sigmoid), -1)

      if detach:
        return (action.detach(), self.std.detach()), self.critic_layer(x).detach()
      else:
        return (action, self.std), self.critic_layer(x)
      
class Value_Model(nn.Module):
    def __init__(self, state_dim):
      super(Value_Model, self).__init__()

      self.nn_layer             = nn.Sequential( nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU() )
      self.critic_layer         = nn.Sequential( nn.Linear(64, 1) )
        
    def forward(self, state, detach = False):
      x   = self.nn_layer(state)

      if detach:
        return self.critic_layer(x).detach()
      else:
        return self.critic_layer(x)
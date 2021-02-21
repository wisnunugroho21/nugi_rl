import torch
import torch.nn as nn
from utils.pytorch_utils import set_device
from model.components.SeperableConv2d import DepthwiseSeparableConv2d

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Policy_Model, self).__init__()

      self.conv1 = nn.Sequential(
        DepthwiseSeparableConv2d(3, 16, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 64, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.conv2 = nn.Sequential(
        DepthwiseSeparableConv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.conv3 = nn.Sequential(
        DepthwiseSeparableConv2d(64, 256, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.state_extractor = nn.Sequential(
        nn.Linear(1, 64),
        nn.ReLU()
      ).float().to(set_device(use_gpu))

      self.nn_layer = nn.Sequential(
        nn.Linear(320, 64),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.actor_tanh_layer = nn.Sequential(
        nn.Linear(64, 1),
        nn.Tanh()
      ).float().to(set_device(use_gpu))

      self.actor_sigmoid_layer = nn.Sequential(
        nn.Linear(64, 2),
        nn.Sigmoid()
      ).float().to(set_device(use_gpu))

      self.critic_layer = nn.Sequential(
        nn.Linear(64, 1)
      ).float().to(set_device(use_gpu))

      self.std = torch.FloatTensor([1.0, 0.5, 0.5]).to(set_device(use_gpu))
        
    def forward(self, datas, detach = False):
      i   = datas[0]
      i   = i.transpose(2, 3).transpose(1, 2)

      i1  = self.conv1(i)
      i2  = self.conv2(i1)
      i3  = self.conv3(i1)
      i23 = i2 + i3
      i4  = i23.mean([2, 3])

      s   = datas[1]
      s1  = self.state_extractor(s)
      x   = torch.cat((s1, i4), -1)
      x   = self.nn_layer(x)

      action_tanh     = self.actor_tanh_layer(x)
      action_sigmoid  = self.actor_sigmoid_layer(x)
      action          = torch.cat((action_tanh, action_sigmoid), -1)

      if detach:
        return (action.detach(), self.std.detach()), self.critic_layer(x).detach()
      else:
        return (action, self.std), self.critic_layer(x)
      
class Value_Model(nn.Module):
    def __init__(self, state_dim, use_gpu = True):
      super(Value_Model, self).__init__()

      self.conv1 = nn.Sequential(
        DepthwiseSeparableConv2d(3, 16, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 64, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.conv2 = nn.Sequential(
        DepthwiseSeparableConv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.conv3 = nn.Sequential(
        DepthwiseSeparableConv2d(64, 256, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.state_extractor = nn.Sequential(
        nn.Linear(1, 64),
        nn.ReLU()
      ).float().to(set_device(use_gpu))

      self.nn_layer = nn.Sequential(
        nn.Linear(320, 64),
        nn.ReLU(),
      ).float().to(set_device(use_gpu))

      self.critic_layer = nn.Sequential(
        nn.Linear(64, 1)
      ).float().to(set_device(use_gpu))
        
    def forward(self, datas, detach = False):
      i   = datas[0]
      i   = i.transpose(2, 3).transpose(1, 2)

      i1  = self.conv1(i)
      i2  = self.conv2(i1)
      i3  = self.conv3(i1)
      i23 = i2 + i3
      i4  = i23.mean([2, 3])

      s   = datas[1]
      s1  = self.state_extractor(s)
      x   = torch.cat((s1, i4), -1)
      x   = self.nn_layer(x)

      if detach:
        return self.critic_layer(x).detach()
      else:
        return self.critic_layer(x)
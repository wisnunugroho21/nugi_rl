import torch.nn as nn

from model.components.SeperableConv2d import DepthwiseSeparableConv2d
from model.components.CMTBlock import CMTBlock
from model.components.RegNet import Stage

class CnnModel(nn.Module):
    def __init__(self):
      super(CnnModel, self).__init__()

      self.conv = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size = 2, stride = 2),
        nn.ReLU(),
        nn.BatchNorm2d(4),
        Stage(4, 8),
        Stage(8, 16),
        Stage(16, 32),
        CMTBlock(32),
        nn.Conv2d(32, 64, kernel_size = 2, stride = 2),
        nn.ReLU(),
        CMTBlock(64),
        nn.Conv2d(64, 128, kernel_size = 2, stride = 2),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 128, kernel_size = 5, stride = 1, padding = 0),
        nn.ReLU(),
        nn.Flatten()
      )
        
    def forward(self, image, detach = False):
      out = self.conv(image)

      if detach:
        return out.detach()
      else:
        return out
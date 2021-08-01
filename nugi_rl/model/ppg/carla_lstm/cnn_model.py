import torch.nn as nn

from model.components.ASPP import AtrousSpatialPyramidConv2d
from model.components.SeperableConv2d import DepthwiseSeparableConv2d
from model.components.Downsampler import Downsampler

class ExtractEncoder(nn.Module):
    def __init__(self, dim):
        super(ExtractEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv2d(dim, dim, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),                    
        )

        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv2d(dim, dim, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = x + x1

        x2 = self.conv2(x1)
        x2 = x1 + x2

        return x2

class CnnModel(nn.Module):
    def __init__(self):
      super(CnnModel, self).__init__()   

      self.conv = nn.Sequential(
        DepthwiseSeparableConv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 32, kernel_size = 8, stride = 4, padding = 2, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(32, 32, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(32, 64, kernel_size = 8, stride = 4, padding = 2, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(64, 128, kernel_size = 8, stride = 4, padding = 2, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 128, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 256, kernel_size = 5, stride = 1, padding = 0, bias = False),
        nn.ReLU(),
      )
        
    def forward(self, image, detach = False):
      n, t = image.shape[0], 0

      if len(image.shape) == 5:
        n, t, c, h, w = image.shape
        image = image.transpose(0, 1).reshape(n * t, c, h, w)

      out = self.conv(image)
      out = out.mean([-1, -2])

      if t > 0:
        out = out.view(t, n, -1)

      if detach:
        return out.detach()
      else:
        return out
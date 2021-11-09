import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
import torch

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, nin: int, nout: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = True, depth_multiplier: int = 1):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        self.nn_layer = nn.Sequential(
            nn.Conv2d(nin, nin * depth_multiplier, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias, groups = nin),
            nn.Conv2d(nin * depth_multiplier, nout, kernel_size = 1, bias = bias)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.nn_layer(x)

class BlockSame(nn.Module):
    def __init__(self, dim: int, b: int = 1, g: int = 1) -> None:
        super(BlockSame, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(dim, math.floor(dim / b), kernel_size = 1, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(math.floor(dim / b)),
            nn.Conv2d(math.floor(dim / b), math.floor(dim / b), kernel_size = 3, stride = 1, padding = 1, groups = g),
            nn.ReLU(),
            nn.BatchNorm2d(math.floor(dim / b)),
            nn.Conv2d(math.floor(dim / b), dim, kernel_size = 1, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv(x)
        return x + x1

class BlockDown(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, b: int = 1, g: int = 1) -> None:
        super(BlockDown, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, math.floor(dim_out / b), kernel_size = 1, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(math.floor(dim_out / b)),
            nn.Conv2d(math.floor(dim_out / b), math.floor(dim_out / b), kernel_size = 4, stride = 2, padding = 1, groups = g),
            nn.ReLU(),
            nn.BatchNorm2d(math.floor(dim_out / b)),
            nn.Conv2d(math.floor(dim_out / b), dim_out, kernel_size = 1, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size = 2, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return x1 + x2

class Stage(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, d:int = 2) -> None:
        super(Stage, self).__init__()

        self.convlist = nn.ModuleList([
            BlockDown(dim_in, dim_out),
        ])

        for _ in range(1, d):
            self.convlist.append(
                BlockSame(dim_out)
            )

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convlist:
            x = conv(x)

        return x
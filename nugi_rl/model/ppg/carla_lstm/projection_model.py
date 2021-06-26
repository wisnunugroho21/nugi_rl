import torch.nn as nn
import torch

from helpers.pytorch_utils import set_device

class ProjectionModel(nn.Module):
    def __init__(self):
      super(ProjectionModel, self).__init__()

      self.nn_layer   = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
      )

    def forward(self, res, detach = False):      
      if detach:
        return self.nn_layer(res).detach()
      else:
        return self.nn_layer(res)
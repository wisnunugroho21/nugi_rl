import torch.nn as nn
from torch import Tensor

from nugi_rl.distribution.base import Distribution

class EntropyLoss(nn.Module):
    def __init__(self, distribution: Distribution, entropy_coef: float = 0.01):
        self.entropy_coef       = entropy_coef
        self.distribution       = distribution

    def forward(self, action_datas: tuple) -> Tensor:
        dist_entropy    = self.distribution.entropy(*action_datas).mean()
        return -1 * self.entropy_coef * dist_entropy
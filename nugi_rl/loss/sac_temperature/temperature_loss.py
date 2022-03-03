import torch
import torch.nn as nn
from torch import Tensor

from nugi_rl.distribution.base import Distribution

class TemperatureLoss(nn.Module):
    def __init__(self, distribution: Distribution, desired_alpha: int = -6):
        super().__init__()
        
        self.distribution   = distribution
        self.desired_alpha  = desired_alpha

    def forward(self, action_datas: tuple, actions: Tensor, alpha: Tensor) -> Tensor:
        log_prob                = self.distribution.logprob(*action_datas, actions).detach()
        policy_loss             = (-alpha * log_prob - alpha * self.desired_alpha).mean()
        return policy_loss
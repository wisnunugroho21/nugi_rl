import torch
import torch.nn as nn
from torch import Tensor

from nugi_rl.distribution.base import Distribution

class PolicyLoss(nn.Module):
    def __init__(self, distribution: Distribution, alpha: int = 0.2):
        super().__init__()
        
        self.distribution   = distribution
        self.alpha          = alpha

    def forward(self, action_datas: tuple, actions: Tensor, q_value1: Tensor, q_value2: Tensor) -> Tensor:
        log_prob                = self.distribution.logprob(*action_datas, actions)
        policy_loss             = (self.alpha * log_prob - torch.min(q_value1, q_value2)).mean()
        return policy_loss
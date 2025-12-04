import torch.nn as nn
from torch import Tensor

from nugi_rl.distribution.base import Distribution


class TemperatureLoss(nn.Module):
    def __init__(self, distribution: Distribution, desired_alpha: float = -6):
        super().__init__()

        self.distribution = distribution
        self.desired_alpha = desired_alpha

    def forward(
        self,
        alpha: Tensor,
        action_datas: Tensor,
        actions: Tensor,
    ) -> Tensor:
        log_prob = self.distribution.logprob(action_datas, actions).detach()
        entropy_loss = (self.desired_alpha - alpha * log_prob).mean()

        return entropy_loss

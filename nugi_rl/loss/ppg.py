import torch.nn as nn
from torch import Tensor

from nugi_rl.distribution.base import Distribution


class PpgLoss(nn.Module):
    def __init__(self, distribution: Distribution):
        super().__init__()

        self.distribution = distribution

    def forward(
        self,
        action_datas: Tensor,
        old_action_datas: Tensor,
        values: Tensor,
        returns: Tensor,
    ) -> Tensor:
        Kl = self.distribution.kldivergence(old_action_datas, action_datas).mean()
        aux_loss = ((returns - values).pow(2) * 0.5).mean()

        return aux_loss + Kl

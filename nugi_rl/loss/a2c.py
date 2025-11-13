import torch.nn as nn
from torch import Tensor

from nugi_rl.distribution.base import Distribution


class A2C(nn.Module):
    def __init__(self, distribution: Distribution):
        super().__init__()
        self.distribution = distribution

    def forward(
        self, action_datas: Tensor, actions: Tensor, advantages: Tensor
    ) -> Tensor:
        logprobs = self.distribution.logprob(action_datas, actions) + 1e-5

        pg_target = logprobs * advantages
        loss = pg_target.mean()

        return loss

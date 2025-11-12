import torch
from torch import Tensor
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from nugi_rl.distribution.base import Distribution


class BasicContinous(Distribution):
    def sample(self, datas: Tensor) -> Tensor:
        mean = datas[0]
        std = datas[1]

        distribution = Normal(torch.zeros_like(mean), torch.ones_like(std))
        rand = distribution.sample()

        return mean + std * rand

    def entropy(self, datas: Tensor) -> Tensor:
        mean = datas[0]
        std = datas[1]

        distribution = Normal(mean, std)
        return distribution.entropy()

    def logprob(self, datas: Tensor, value: Tensor) -> Tensor:
        mean = datas[0]
        std = datas[1]

        distribution = Normal(mean, std)
        return distribution.log_prob(value)

    def kldivergence(self, datas1: Tensor, datas2: Tensor) -> Tensor:
        mean1 = datas1[0]
        std1 = datas1[1]

        mean2 = datas2[0]
        std2 = datas2[1]

        distribution1 = Normal(mean1, std1)
        distribution2 = Normal(mean2, std2)
        return kl_divergence(distribution1, distribution2)

    def deterministic(self, data: Tensor) -> Tensor:
        return data.squeeze(0)

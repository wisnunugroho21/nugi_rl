import torch
from torch import Tensor
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from nugi_rl.distribution.base import Distribution

class BasicContinous(Distribution):
    def sample(self, datas: tuple) -> Tensor:
        mean, std = datas

        distribution    = Normal(torch.zeros_like(mean), torch.ones_like(std))
        rand            = distribution.sample()
        return mean + std * rand
        
    def entropy(self, datas: tuple) -> Tensor:
        mean, std = datas
        
        distribution = Normal(mean, std)
        return distribution.entropy()
        
    def logprob(self, datas: tuple, value_data: Tensor) -> Tensor:
        mean, std = datas

        distribution = Normal(mean, std)
        return distribution.log_prob(value_data)

    def kldivergence(self, datas1: tuple, datas2: tuple) -> Tensor:
        mean1, std1 = datas1
        mean2, std2 = datas2

        distribution1 = Normal(mean1, std1)
        distribution2 = Normal(mean2, std2)
        return kl_divergence(distribution1, distribution2)

    def deterministic(self, datas: tuple) -> Tensor:
        mean, _ = datas
        return mean.squeeze(0)
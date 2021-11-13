import torch
from torch import Tensor
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from nugi_rl.distribution.base import Distribution

class BasicContinous(Distribution):
    def sample(self, mean: Tensor, std: Tensor) -> Tensor:
        distribution    = Normal(torch.zeros_like(mean), torch.ones_like(std))
        rand            = distribution.sample()
        return mean + std * rand
        
    def entropy(self, mean: Tensor, std: Tensor) -> Tensor:        
        distribution = Normal(mean, std)
        return distribution.entropy()
        
    def logprob(self, mean: Tensor, std: Tensor, value_data: Tensor) -> Tensor:
        distribution = Normal(mean, std)
        return distribution.log_prob(value_data)

    def kldivergence(self, mean1: Tensor, std1: Tensor, mean2: Tensor, std2: Tensor) -> Tensor:
        distribution1 = Normal(mean1, std1)
        distribution2 = Normal(mean2, std2)
        return kl_divergence(distribution1, distribution2)

    def deterministic(self, mean: Tensor, std: Tensor) -> Tensor:
        return mean.squeeze(0)
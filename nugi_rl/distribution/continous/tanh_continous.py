import torch
from torch import Tensor
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from nugi_rl.distribution.continous.basic_continous import BasicContinous

class TanhContinous(BasicContinous):
    def sample(self, mean: Tensor, std: Tensor, onlyTanh: bool = False, onlySampled: bool = True) -> Tensor:
        distribution    = Normal(torch.zeros_like(mean), torch.ones_like(std))
        rand            = distribution.sample()

        sampled = mean + std * rand
        result  = sampled.tanh()

        if onlyTanh:
            return result
        elif onlySampled:
            return sampled
        else:
            return result, sampled
        
    def entropy(self, mean: Tensor, std: Tensor) -> Tensor:        
        distribution    = Normal(mean, std)
        
        rand            = distribution.sample()
        log_det_jacob   = (1 - rand.square()).log().sum()

        return distribution.entropy() + log_det_jacob
        
    def logprob(self, mean: Tensor, std: Tensor, value_data: Tensor) -> Tensor:
        distribution    = Normal(mean, std)
        log_det_jacob   = (1 - value_data.square()).log().sum()

        return distribution.log_prob(value_data) - log_det_jacob

    def kldivergence(self, mean1: Tensor, std1: Tensor, mean2: Tensor, std2: Tensor) -> Tensor:
        distribution1   = Normal(mean1, std1)
        distribution2   = Normal(mean2, std2)

        rand1           = distribution1.sample()
        rand2           = distribution2.sample()

        log_det_jacob1  = (1 - rand1.square()).log().sum()
        log_det_jacob2  = (1 - rand2.square()).log().sum()

        return kl_divergence(distribution1, distribution2) + log_det_jacob2 - log_det_jacob1

    def deterministic(self, mean: Tensor, std: Tensor) -> Tensor:
        return mean.squeeze(0)
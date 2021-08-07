from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import torch

from helpers.pytorch_utils import set_device, to_list

class BasicContinous():
    def sample(self, datas):
        mean, std = datas

        distribution    = Normal(torch.zeros_like(mean), torch.ones_like(std))
        rand            = distribution.sample()
        return mean + std * rand
        
    def entropy(self, datas):
        mean, std = datas
        
        distribution = Normal(mean, std)
        return distribution.entropy()
        
    def logprob(self, datas, value_data):
        mean, std = datas

        distribution = Normal(mean, std)
        return distribution.log_prob(value_data)

    def kldivergence(self, datas1, datas2):
        mean1, std1 = datas1
        mean2, std2 = datas2

        distribution1 = Normal(mean1, std1)
        distribution2 = Normal(mean2, std2)
        return kl_divergence(distribution1, distribution2)

    def deterministic(self, datas):
        mean, _ = datas
        return mean.squeeze(0)
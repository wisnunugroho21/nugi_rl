import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

from nugi_rl.distribution.base import Distribution

class MultivariateContinous(Distribution):
    def sample(self, mean: Tensor, std: Tensor) -> Tensor:
        std = torch.diag_embed(std)

        distribution    = MultivariateNormal(torch.zeros_like(mean), torch.ones_like(std))
        rand            = distribution.sample().squeeze(0)
        return mean + std * rand
        
    def entropy(self, mean: Tensor, std: Tensor) -> Tensor:
        std = torch.diag_embed(std)

        distribution = MultivariateNormal(mean, std) 
        return distribution.entropy()
        
    def logprob(self, mean: Tensor, std: Tensor, value_data: Tensor) -> Tensor:
        std = torch.diag_embed(std)

        distribution = MultivariateNormal(mean, std)
        return distribution.log_prob(value_data)

    def kldivergence(self, mean1: Tensor, std1: Tensor, mean2: Tensor, std2: Tensor) -> Tensor:
        std1 = torch.diag_embed(std1)
        std2 = torch.diag_embed(std2)

        distribution1 = MultivariateNormal(mean1, std1)
        distribution2 = MultivariateNormal(mean2, std2)
        return kl_divergence(distribution1, distribution2)

    def kldivergence_mean(self, mean1: Tensor, std1: Tensor, mean2: Tensor, std2: Tensor) -> Tensor:
        mean1, mean2    = mean1.unsqueeze(-1), mean2.unsqueeze(-1)
        std1            = torch.diag_embed(std1)       

        Kl_mean = 0.5 * (mean2 - mean1).transpose(-2, -1) @ std1.inverse() @ (mean2 - mean1)
        return Kl_mean

    def kldivergence_cov(self, mean1: Tensor, std1: Tensor, mean2: Tensor, std2: Tensor) -> Tensor:
        d               = mean1.shape[-1]
        std1, std2      = torch.diag_embed(std1), torch.diag_embed(std2)

        Kl_cov  = 0.5 * ((std2.inverse() @ std1).diagonal(dim1 = -2, dim2 = -1).sum(-1) - d + (torch.linalg.det(std2) / (torch.linalg.det(std1) + 1e-3)).log())
        return Kl_cov

    def deterministic(self, mean: Tensor, std: Tensor) -> Tensor:
        return mean.squeeze(0)
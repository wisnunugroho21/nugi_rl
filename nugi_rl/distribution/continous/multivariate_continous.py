import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

from nugi_rl.distribution.continous.basic_continous import BasicContinous

class MultivariateContinous(BasicContinous):
    def sample(self, datas: tuple) -> Tensor:
        mean, std = datas
        std = torch.diag_embed(std)

        distribution    = MultivariateNormal(mean, std)
        action          = distribution.sample().squeeze(0)
        return action
        
    def entropy(self, datas: tuple) -> Tensor:
        mean, std = datas
        std = torch.diag_embed(std)

        distribution = MultivariateNormal(mean, std) 
        return distribution.entropy()
        
    def logprob(self, datas: tuple, value_data: Tensor) -> Tensor:
        mean, std = datas
        std = torch.diag_embed(std)

        distribution = MultivariateNormal(mean, std)
        return distribution.log_prob(value_data)

    def kldivergence(self, datas1: tuple, datas2: tuple) -> Tensor:
        mean1, std1 = datas1
        mean2, std2 = datas2

        std1 = torch.diag_embed(std1)
        std2 = torch.diag_embed(std2)

        distribution1 = MultivariateNormal(mean1, std1)
        distribution2 = MultivariateNormal(mean2, std2)
        return kl_divergence(distribution1, distribution2)

    def kldivergence_mean(self, datas1: tuple, datas2: tuple) -> Tensor:
        mean1, cov1     = datas1
        mean2, _        = datas2

        mean1, mean2    = mean1.unsqueeze(-1), mean2.unsqueeze(-1)
        cov1            = torch.diag_embed(cov1)       

        Kl_mean = 0.5 * (mean2 - mean1).transpose(-2, -1) @ cov1.inverse() @ (mean2 - mean1)
        return Kl_mean

    def kldivergence_cov(self, datas1: tuple, datas2: tuple) -> Tensor:
        mean1, cov1     = datas1
        _, cov2         = datas2

        d               = mean1.shape[-1]
        cov1, cov2      = torch.diag_embed(cov1), torch.diag_embed(cov2)

        Kl_cov  = 0.5 * ((cov2.inverse() @ cov1).diagonal(dim1 = -2, dim2 = -1).sum(-1) - d + (torch.linalg.det(cov2) / (torch.linalg.det(cov1) + 1e-3)).log())
        return Kl_cov

    def deterministic(self, datas: tuple) -> Tensor:
        mean, _ = datas
        return mean.squeeze(0)
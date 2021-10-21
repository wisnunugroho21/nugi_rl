import torch
from torch import Tensor

from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

from nugi_rl.distribution.base import Distribution

class BasicDiscrete(Distribution):
    def sample(self, datas: Tensor) -> Tensor:
        distribution = Categorical(datas)
        return distribution.sample().int()
        
    def entropy(self, datas: Tensor) -> Tensor:
        distribution = Categorical(datas)
        return distribution.entropy().unsqueeze(1)
        
    def logprob(self, datas: Tensor, value_data: Tensor) -> Tensor:
        distribution = Categorical(datas)        
        return distribution.log_prob(value_data).unsqueeze(1)

    def kldivergence(self, datas1: Tensor, datas2: Tensor) -> Tensor:
        distribution1 = Categorical(datas1)
        distribution2 = Categorical(datas2)
        return kl_divergence(distribution1, distribution2).unsqueeze(1)

    def deterministic(self, datas: Tensor) -> Tensor:
        return torch.argmax(datas, 1).int()

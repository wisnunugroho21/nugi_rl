import torch
from torch import Tensor

from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

from nugi_rl.distribution.base import Distribution

class BasicDiscrete(Distribution):
    def sample(self, logits: Tensor) -> Tensor:
        distribution = Categorical(logits)
        return distribution.sample().int()
        
    def entropy(self, logits: Tensor) -> Tensor:
        distribution = Categorical(logits)
        return distribution.entropy().unsqueeze(1)
        
    def logprob(self, logits: Tensor, value: Tensor) -> Tensor:
        distribution = Categorical(logits)        
        return distribution.log_prob(value).unsqueeze(1)

    def kldivergence(self, logits1: Tensor, logits2: Tensor) -> Tensor:
        distribution1 = Categorical(logits1)
        distribution2 = Categorical(logits2)
        return kl_divergence(distribution1, distribution2).unsqueeze(1)

    def deterministic(self, logits: Tensor) -> Tensor:
        return torch.argmax(logits, 1).int()

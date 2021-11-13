import torch
from torch import Tensor

from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

from nugi_rl.distribution.base import Distribution

class Ordinal(Distribution):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device

    def _compute_ordinal(self, logits: Tensor) -> Tensor:      
        logits = logits.unsqueeze(-1)

        a1 = torch.ones(logits.size(2), logits.size(2)).to(self.device).triu().transpose(0, 1).repeat(logits.size(0), logits.size(1), 1, 1)
        a2 = a1.logical_not().float()

        a3 = logits.log()
        a4 = (1 - logits).log()

        out = torch.matmul(a1, a3) + torch.matmul(a2, a4)
        out = torch.nn.functional.softmax(out, dim = 2)
        out = out.squeeze(-1)        
        
        return out

    def sample(self, logits: Tensor) -> Tensor:
        logits = self._compute_ordinal(logits)
        
        distribution = Categorical(logits)
        return distribution.sample().int()
        
    def entropy(self, logits: Tensor) -> Tensor:
        logits = self._compute_ordinal(logits)
        
        distribution = Categorical(logits)
        return distribution.entropy()
        
    def logprob(self, logits: Tensor, value: Tensor) -> Tensor:
        logits = self._compute_ordinal(logits)

        distribution = Categorical(logits)        
        return distribution.log_prob(value)

    def kldivergence(self, logits1: Tensor, logits2: Tensor) -> Tensor:
        logits1 = self._compute_ordinal(logits1)
        logits2 = self._compute_ordinal(logits2)

        distribution1 = Categorical(logits1)
        distribution2 = Categorical(logits2)
        return kl_divergence(distribution1, distribution2)

    def deterministic(self, logits: Tensor) -> Tensor:
        logits = self._compute_ordinal(logits)
        return torch.argmax(logits, 1).int()

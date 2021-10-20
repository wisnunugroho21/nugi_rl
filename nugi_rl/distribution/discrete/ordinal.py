import torch
from torch import Tensor

from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

from nugi_rl.distribution.base import Distribution

class Ordinal(Distribution):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device

    def _compute_ordinal(self, datas: Tensor) -> Tensor:        
        a1 = torch.ones(datas.size(-1), datas.size(-1)).to(self.device).triu().transpose(0, 1).repeat(datas.size(0), datas.size(1), 1, 1)
        a2 = a1.logical_not().float()

        datas = datas.unsqueeze(-1)

        a3 = datas.log()
        a4 = (1 - datas).log()

        out = torch.matmul(a1, a3) + torch.matmul(a2, a4)
        out = out.squeeze(-1)        
        out = torch.nn.functional.softmax(out, dim = -1)
        
        return out

    def sample(self, datas: Tensor) -> Tensor:
        datas = self._compute_ordinal(datas)
        
        distribution = Categorical(datas)
        return distribution.sample().int()
        
    def entropy(self, datas: Tensor) -> Tensor:
        datas = self._compute_ordinal(datas)
        
        distribution = Categorical(datas)
        return distribution.entropy()
        
    def logprob(self, datas: Tensor, value_data: Tensor) -> Tensor:
        datas = self._compute_ordinal(datas)

        distribution = Categorical(datas)        
        return distribution.log_prob(value_data)

    def kldivergence(self, datas1: Tensor, datas2: Tensor) -> Tensor:
        datas1 = self._compute_ordinal(datas1)
        datas2 = self._compute_ordinal(datas2)

        distribution1 = Categorical(datas1)
        distribution2 = Categorical(datas2)
        return kl_divergence(distribution1, distribution2)

    def deterministic(self, datas: Tensor) -> Tensor:
        datas = self._compute_ordinal(datas)
        return torch.argmax(datas, 1).int()

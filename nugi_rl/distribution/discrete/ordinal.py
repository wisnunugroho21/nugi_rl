import torch
from torch import Tensor

from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

from nugi_rl.distribution.discrete.basic_discrete import BasicDiscrete

class Ordinal(BasicDiscrete):
    def __init__(self, bins = 15) -> None:
        super().__init__()

        self.act = torch.arange(bins)
        self.act = (2 * self.act) / (bins - 1) - 1

    def _compute_ordinal(self, datas: Tensor) -> Tensor:
        a1 = torch.ones(datas.size(-1), datas.size(-1)).triu().transpose(0, 1).repeat(datas.size(0), 1, 1)
        a2 = a1.logical_not().float()

        datas = datas.unsqueeze(-1)

        a3 = datas.log()
        a4 = (1 - datas).log()

        return torch.matmul(a1, a3) + torch.matmul(a2, a4)

    def sample(self, datas: Tensor) -> Tensor:
        datas = self._compute_ordinal(datas)
        return super().sample(datas)
        
    def entropy(self, datas: Tensor) -> Tensor:
        datas = self._compute_ordinal(datas)
        return super().entropy(datas)
        
    def logprob(self, datas: Tensor, value_data: Tensor) -> Tensor:
        datas = self._compute_ordinal(datas)  
        return super().logprob(datas, value_data)

    def kldivergence(self, datas1: Tensor, datas2: Tensor) -> Tensor:
        datas1 = self._compute_ordinal(datas1)
        datas2 = self._compute_ordinal(datas2)

        super().kldivergence(datas1, datas2)

    def deterministic(self, datas: Tensor) -> Tensor:
        datas = self._compute_ordinal(datas)
        super().deterministic(datas)

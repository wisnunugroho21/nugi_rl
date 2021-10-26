import torch
import math

from torch import Tensor

from nugi_rl.distribution.base import Distribution

class PhiLoss():
    def __init__(self, distribution: Distribution):
        self.distribution       = distribution

    def compute_loss(self, action_datas: tuple, actions: Tensor, temperature: Tensor, advantages: Tensor) -> Tensor:
        temperature         = temperature.detach()
        top_adv, top_idx    = torch.topk(advantages, math.ceil(advantages.size(0) / 2), 0)

        logprobs            = self.distribution.logprob(action_datas, actions)
        top_logprobs        = logprobs[top_idx]        

        ratio               = top_adv / (temperature + 1e-3)
        psi                 = torch.nn.functional.softmax(ratio, dim = 0)

        loss                = -1 * (psi * top_logprobs).sum()
        return loss
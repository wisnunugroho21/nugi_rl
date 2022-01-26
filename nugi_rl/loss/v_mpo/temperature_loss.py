import torch
import torch.nn as nn
import math

from torch import Tensor

class TemperatureLoss(nn.Module):
    def __init__(self, coef_temp: int = 0.0001):
        super().__init__()
        self.coef_temp          = coef_temp

    def forward(self, temperature: Tensor, advantages: Tensor) -> Tensor:
        top_adv, _  = torch.topk(advantages, math.ceil(advantages.size(0) / 2), 0)

        ratio       = top_adv / (temperature + 1e-3)
        n           = torch.tensor(top_adv.size(0))

        loss        = temperature * self.coef_temp + temperature * (torch.logsumexp(ratio, dim = 0) - n.log())
        return loss.squeeze()
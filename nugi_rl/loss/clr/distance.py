import torch
from torch import Tensor

from nugi_rl.loss.clr.base import CLR

class DistancesClr(CLR):
    def compute_loss(self, first_encoded: Tensor, second_encoded: Tensor) -> Tensor:
        return torch.nn.functional.pairwise_distance(second_encoded, first_encoded).mean()
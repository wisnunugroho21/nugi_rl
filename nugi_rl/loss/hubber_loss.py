import torch
import torch.nn as nn
from torch import Tensor


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = 1.0

    def forward(
        self,
        predicted_value: Tensor,
        target_value: Tensor,
    ) -> Tensor:
        alpha = target_value - predicted_value

        l2_loss = 0.5 * alpha.square()
        l1_loss = self.delta * (alpha.abs() - 0.5 * self.delta)
        loss = torch.where(alpha.abs() < self.delta, l2_loss, l1_loss)

        return loss.mean()

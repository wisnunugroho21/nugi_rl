import torch.nn as nn
from torch import Tensor

from nugi_rl.loss.hubber_loss import HuberLoss


class ValueLoss(nn.Module):
    def __init__(self, value_loss_coef: float = 1.0, value_clip: float | None = None):
        super().__init__()

        self.value_clip = value_clip
        self.value_loss_coef = value_loss_coef
        self.huber_loss = HuberLoss()

    def forward(
        self, values: Tensor, returns: Tensor, old_values: Tensor | None = None
    ) -> Tensor:
        if self.value_clip is None or old_values is None:
            loss = self.huber_loss(values, returns)
        else:
            old_values = old_values.detach()
            vpredclipped = old_values + (values - old_values).clamp(
                -self.value_clip, self.value_clip
            )

            loss = self.huber_loss(vpredclipped, returns)

        return self.value_loss_coef * loss

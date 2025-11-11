import torch
import torch.nn as nn
from torch import Tensor


class ValueLoss(nn.Module):
    def __init__(self, value_loss_coef: float = 1.0, value_clip: float | None = None):
        super().__init__()

        self.value_clip = value_clip
        self.value_loss_coef = value_loss_coef

    def forward(
        self, values: Tensor, advantages: Tensor, old_values: Tensor | None = None
    ) -> Tensor:
        returns = (advantages + values).detach()

        if self.value_clip is None or old_values is None:
            loss = ((returns - values).pow(2) * 0.5).mean()
        else:
            old_values = old_values.detach()

            vpredclipped = old_values + torch.clamp(
                values - old_values, -self.value_clip, self.value_clip
            )
            loss = ((returns - vpredclipped).pow(2) * 0.5).mean()

        return self.value_loss_coef * loss

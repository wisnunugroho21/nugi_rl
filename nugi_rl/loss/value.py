import torch
from torch import Tensor

class ValueLoss():
    def __init__(self, vf_loss_coef: float = 1.0, value_clip: float = None):
        self.value_clip         = value_clip
        self.vf_loss_coef       = vf_loss_coef
 
    def compute_loss(self, values: Tensor, advantages: Tensor, old_values: Tensor = None) -> Tensor:
        returns = (advantages + values).detach()

        if self.value_clip is None or old_values is None:
            loss            = ((returns - values).pow(2) * 0.5).mean()
        else:
            vpredclipped    = old_values + torch.clamp(values - old_values, -self.value_clip, self.value_clip)
            loss            = ((returns - vpredclipped).pow(2) * 0.5).mean()

        return self.vf_loss_coef * loss
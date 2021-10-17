import torch
from torch import Tensor

from nugi_rl.policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from nugi_rl.loss.value.base import ValueLoss

class ValueLoss(ValueLoss):
    def __init__(self, advantage_function: GeneralizedAdvantageEstimation, value_clip: float = 1.0, vf_loss_coef: float = 1.0):
        self.value_clip         = value_clip
        self.vf_loss_coef       = vf_loss_coef
        self.advantage_function = advantage_function
 
    def compute_loss(self, values: Tensor, old_values: Tensor, next_values: Tensor, rewards: Tensor, dones: Tensor) -> Tensor:
        advantages  = self.advantage_function.compute_advantages(rewards, values, next_values, dones).detach()
        returns     = (advantages + values).detach()

        if self.value_clip is None:
            loss            = ((returns - values).pow(2) * 0.5).mean()
        else:
            vpredclipped    = old_values + torch.clamp(values - old_values, -self.value_clip, self.value_clip)
            loss            = ((returns - vpredclipped).pow(2) * 0.5).mean()

        return loss * self.vf_loss_coef
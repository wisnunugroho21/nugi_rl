import torch
import math

from torch import Tensor

from nugi_rl.policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation

class TemperatureLoss():
    def __init__(self, advantage_function: GeneralizedAdvantageEstimation, coef_temp: int = 0.0001):
        self.advantage_function = advantage_function
        self.coef_temp          = coef_temp

    def compute_loss(self, values: Tensor, next_values: Tensor, rewards: Tensor, dones: Tensor, temperature: Tensor) -> Tensor:
        advantages  = self.advantage_function.compute_advantages(rewards, values, next_values, dones).detach()                
        top_adv, _  = torch.topk(advantages, math.ceil(advantages.size(0) / 2), 0)

        ratio       = top_adv / (temperature + 1e-6)
        n           = torch.tensor(top_adv.size(0))

        loss        = temperature * self.coef_temp + temperature * (torch.logsumexp(ratio, dim = 0) - n.log())
        return loss.squeeze()
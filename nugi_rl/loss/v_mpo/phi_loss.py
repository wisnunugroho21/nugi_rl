import torch
import math

from torch import Tensor

from nugi_rl.distribution.base import Distribution
from nugi_rl.policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation

class PhiLoss():
    def __init__(self, distribution: Distribution, advantage_function: GeneralizedAdvantageEstimation):
        self.advantage_function = advantage_function
        self.distribution       = distribution

    def compute_loss(self, action_datas: tuple, values: Tensor, next_values: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor, temperature: Tensor) -> Tensor:
        temperature         = temperature.detach()

        advantages          = self.advantage_function.compute_advantages(rewards, values, next_values, dones).detach()
        top_adv, top_idx    = torch.topk(advantages, math.ceil(advantages.size(0) / 2), 0)

        logprobs            = self.distribution.logprob(action_datas, actions)
        top_logprobs        = logprobs[top_idx]        

        ratio               = top_adv / (temperature + 1e-3)
        psi                 = torch.nn.functional.softmax(ratio, dim = 0)

        loss                = -1 * (psi * top_logprobs).sum()
        return loss
import torch
from torch import Tensor

from nugi_rl.distribution.base import Distribution
from nugi_rl.policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from nugi_rl.loss.ppo.base import Ppo

class PpoClip(Ppo):
    def __init__(self, distribution: Distribution, advantage_function: GeneralizedAdvantageEstimation, 
        policy_clip: float = 0.2):

        self.policy_clip        = policy_clip

        self.advantage_function = advantage_function
        self.distribution       = distribution
 
    def compute_loss(self, action_datas: tuple, old_action_datas: tuple, values: Tensor, next_values: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor) -> Tensor:
        advantages      = self.advantage_function.compute_advantages(rewards, values, next_values, dones).detach()

        logprobs        = self.distribution.logprob(action_datas, actions) + 1e-5
        old_logprobs    = (self.distribution.logprob(old_action_datas, actions) + 1e-5).detach()

        ratios          = (logprobs - old_logprobs).exp() 
        surr1           = ratios * advantages
        surr2           = ratios.clamp(1 - self.policy_clip, 1 + self.policy_clip) * advantages
        loss            = torch.min(surr1, surr2).mean()

        return -1 * loss
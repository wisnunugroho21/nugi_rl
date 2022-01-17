import torch
from torch import Tensor

from nugi_rl.distribution.base import Distribution
from nugi_rl.loss.ppo.base import Ppo

class TrulyPpo(Ppo):
    def __init__(self, distribution: Distribution, policy_kl_range: float = 0.0008, policy_params: float = 20):
        self.policy_kl_range    = policy_kl_range
        self.policy_params      = policy_params
        self.distribution       = distribution

    def forward(self, action_datas: tuple, old_action_datas: tuple, actions: Tensor, advantages: Tensor) -> Tensor:
        logprobs        = self.distribution.logprob(*action_datas, actions) + 1e-6
        old_logprobs    = (self.distribution.logprob(*old_action_datas, actions) + 1e-6).detach()

        ratios          = (logprobs - old_logprobs).exp()       
        Kl              = self.distribution.kldivergence(*old_action_datas, *action_datas)

        pg_targets  = torch.where(
            (Kl >= self.policy_kl_range) & (ratios > 1.0),
            ratios * advantages - self.policy_params * Kl,
            ratios * advantages
        )
        
        return -1 * pg_targets.mean()
import torch
from torch import Tensor

from nugi_rl.distribution.base import Distribution
from nugi_rl.loss.ppo.base import Ppo

class PpoKl(Ppo):
    def __init__(self, distribution: Distribution, policy_params: float = 1):
        self.policy_params  = policy_params
        self.distribution       = distribution
 
    def compute_loss(self, action_datas: tuple, old_action_datas: tuple, actions: Tensor, advantages: Tensor) -> Tensor:
        logprobs        = self.distribution.logprob(action_datas, actions) + 1e-5
        old_logprobs    = (self.distribution.logprob(old_action_datas, actions) + 1e-5).detach()

        ratios          = (logprobs - old_logprobs).exp()
        Kl              = self.distribution.kldivergence(old_action_datas, action_datas)

        pg_target       = ratios * advantages - self.policy_params * Kl
        loss            = pg_target.mean()

        return -1 * loss
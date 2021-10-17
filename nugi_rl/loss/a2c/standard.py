import torch
from torch import Tensor

from nugi_rl.distribution.base import Distribution
from nugi_rl.policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from nugi_rl.loss.a2c.base import A2C

class A2C(A2C):
    def __init__(self, distribution: Distribution, advantage_function: GeneralizedAdvantageEstimation):
        self.advantage_function = advantage_function
        self.distribution       = distribution
 
    def compute_loss(self, action_datas: tuple, values: Tensor, next_values: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor) -> Tensor:
        advantages      = self.advantage_function.compute_advantages(rewards, values, next_values, dones).detach()
        logprobs        = self.distribution.logprob(action_datas, actions) + 1e-5

        pg_target       = logprobs * advantages
        loss            = pg_target.mean()

        return -1 * loss
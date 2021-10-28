import torch
from torch import Tensor

from nugi_rl.distribution.base import Distribution

class A2C():
    def __init__(self, distribution: Distribution):
        self.distribution       = distribution
 
    def compute_loss(self, action_datas: tuple, actions: Tensor, advantages: Tensor) -> Tensor:
        logprobs        = self.distribution.logprob(action_datas, actions) + 1e-5

        pg_target       = logprobs * advantages
        loss            = pg_target.mean()

        return -1 * loss
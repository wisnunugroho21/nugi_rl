import torch
from torch import Tensor

class PolicyLoss():
    def __init__(self, distribution, alpha = 0.2):
        self.distribution   = distribution
        self.alpha          = alpha

    def compute_loss(self, action_datas: tuple, actions: Tensor, q_value1: Tensor, q_value2: Tensor) -> Tensor:
        log_prob                = self.distribution.logprob(action_datas, actions)
        policy_loss             = (self.alpha * log_prob - torch.min(q_value1, q_value2)).mean()
        return policy_loss
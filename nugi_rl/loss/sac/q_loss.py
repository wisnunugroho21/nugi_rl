import torch
from torch import Tensor

class QLoss():
    def __init__(self, distribution, gamma = 0.99, alpha = 0.2):
        self.gamma          = gamma
        self.distribution   = distribution
        self.alpha          = alpha

    def compute_loss(self, predicted_q1: Tensor, predicted_q2: Tensor, target_next_q1: Tensor, target_next_q2: Tensor, next_action_datas: tuple, next_actions: Tensor, reward: Tensor, done: Tensor) -> Tensor:
        log_prob                = self.distribution.logprob(next_action_datas, next_actions)

        target_value            = (torch.min(target_next_q1, target_next_q2) - self.alpha * log_prob)
        target_q_value          = (reward + (1 - done) * self.gamma * target_value).detach()

        q_value_loss1           = ((target_q_value - predicted_q1).pow(2) * 0.5).mean()
        q_value_loss2           = ((target_q_value - predicted_q2).pow(2) * 0.5).mean()
        
        return q_value_loss1 + q_value_loss2
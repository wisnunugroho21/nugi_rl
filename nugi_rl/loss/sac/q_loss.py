import torch
import torch.nn as nn
from torch import Tensor

from nugi_rl.distribution.base import Distribution


class QLoss(nn.Module):
    def __init__(
        self, distribution: Distribution, gamma: float = 0.99, alpha: float = 0.2
    ):
        super().__init__()

        self.gamma = gamma
        self.distribution = distribution

    def forward(
        self,
        predicted_q1: Tensor,
        predicted_q2: Tensor,
        target_next_q1: Tensor,
        target_next_q2: Tensor,
        next_action_datas: Tensor,
        next_actions: Tensor,
        reward: Tensor,
        done: Tensor,
        alpha: Tensor,
    ) -> Tensor:
        log_prob = self.distribution.logprob(next_action_datas, next_actions)

        target_value = torch.min(target_next_q1, target_next_q2) - alpha * log_prob
        target_q_value = (reward + self.gamma * (1 - done) * target_value).detach()

        q_value_loss1 = (target_q_value - predicted_q1).pow(2).mean()
        q_value_loss2 = (target_q_value - predicted_q2).pow(2).mean()

        return q_value_loss1 + q_value_loss2

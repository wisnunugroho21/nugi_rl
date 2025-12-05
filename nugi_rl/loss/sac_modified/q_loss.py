import torch
from torch import Tensor

from nugi_rl.distribution.base import Distribution
from nugi_rl.loss.hubber_loss import HuberLoss
from nugi_rl.loss.sac.q_loss import QLoss


class ModifiedQLoss(QLoss):
    def __init__(self, distribution: Distribution, gamma: float = 0.99):
        super().__init__(distribution, gamma)

        self.gamma = gamma
        self.distribution = distribution
        self.huber_loss = HuberLoss()

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
        target_value = torch.min(target_next_q1, target_next_q2)
        target_q_value = (reward + self.gamma * (1 - done) * target_value).detach()

        q1_loss = self.huber_loss(predicted_q1, target_q_value)
        q2_loss = self.huber_loss(predicted_q2, target_q_value)

        return q1_loss + q2_loss

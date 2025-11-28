import torch
import torch.nn as nn
from torch import Tensor

from nugi_rl.loss.hubber_loss import HuberLoss


class QLoss(nn.Module):
    def __init__(self, gamma=0.95):
        super().__init__()
        self.gamma = gamma
        self.huber_loss = HuberLoss()

    def forward(
        self,
        predicted_q1: Tensor,
        predicted_q2: Tensor,
        target_next_q1: Tensor,
        target_next_q2: Tensor,
        reward: Tensor,
        done: Tensor,
    ) -> Tensor:
        target_value = torch.min(target_next_q1, target_next_q2)
        target_q_value = (reward + self.gamma * (1.0 - done) * target_value).detach()

        q1_loss = self.huber_loss(predicted_q1, target_q_value)
        q2_loss = self.huber_loss(predicted_q2, target_q_value)

        return q1_loss + q2_loss

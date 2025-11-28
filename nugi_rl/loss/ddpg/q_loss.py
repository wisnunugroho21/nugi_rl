import torch.nn as nn
from torch import Tensor

from nugi_rl.loss.hubber_loss import HuberLoss


class QLoss(nn.Module):
    def __init__(self, gamma: float = 0.99):
        super().__init__()

        self.gamma = gamma
        self.huber_loss = HuberLoss()

    def forward(
        self,
        predicted_q_value: Tensor,
        target_next_q: Tensor,
        reward: Tensor,
        done: Tensor,
    ) -> Tensor:
        target_q_value = (reward + self.gamma * (1 - done) * target_next_q).detach()
        q_value_loss = self.huber_loss(predicted_q_value, target_q_value)

        return q_value_loss

import torch.nn as nn
from torch import Tensor


class QLoss(nn.Module):
    def __init__(self, gamma: float = 0.99):
        super().__init__()

        self.gamma = gamma

    def forward(
        self,
        predicted_q_value: Tensor,
        target_next_q: Tensor,
        reward: Tensor,
        done: Tensor,
    ) -> Tensor:
        target_q_value = (reward + self.gamma * (1 - done) * target_next_q).detach()
        q_value_loss = (target_q_value - predicted_q_value).pow(2).mean()

        return q_value_loss

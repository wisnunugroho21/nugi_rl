import torch.nn as nn
from torch import Tensor


class TemporalDifference(nn.Module):
    def __init__(self, gamma=0.99):
        super().__init__()

        self.gamma = gamma

    def forward(self, reward: Tensor, next_value: Tensor, done: Tensor) -> Tensor:
        q_values = reward + (1.0 - done) * self.gamma * next_value
        return q_values

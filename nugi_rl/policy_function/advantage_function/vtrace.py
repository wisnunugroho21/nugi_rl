import torch
import torch.nn as nn
from torch import Tensor


class VtraceAdvantageEstimation(nn.Module):
    def __init__(self, gamma=0.99):
        super().__init__()

        self.gamma = gamma

    def forward(
        self,
        rewards: Tensor,
        values: Tensor,
        next_values: Tensor,
        dones: Tensor,
        worker_logprobs: Tensor,
        learner_logprobs: Tensor,
    ) -> Tensor:
        gae: Tensor = torch.zeros(size=[1])
        adv: list[Tensor] = []

        limit = torch.FloatTensor([1.0])
        ratio = torch.min(limit, (learner_logprobs - worker_logprobs).exp())

        delta = rewards + (1.0 - dones) * self.gamma * next_values
        delta = ratio * delta

        for step in reversed(range(len(rewards))):
            gae = (1.0 - dones[step]) * (1.0 - self.gamma) * gae
            gae = delta[step] + ratio * gae
            adv.insert(0, gae)

        return torch.stack(adv) - values

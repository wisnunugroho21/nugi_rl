import torch
from torch import Tensor

class VtraceAdvantageEstimation():
    def __init__(self, gamma = 0.99):
        self.gamma  = gamma

    def compute_advantages(self, rewards: Tensor, values: Tensor, next_values: Tensor, dones: Tensor, worker_logprobs: Tensor, learner_logprobs: Tensor) -> Tensor:
        gae     = 0
        adv     = []

        limit   = torch.FloatTensor([1.0])
        ratio   = torch.min(limit, (learner_logprobs - worker_logprobs).exp())

        delta   = rewards + (1.0 - dones) * self.gamma * next_values - values
        delta   = ratio * delta

        for step in reversed(range(len(rewards))):
            gae   = (1.0 - dones[step]) * (1.0 - self.gamma) * gae
            gae   = delta[step] + ratio * gae
            adv.insert(0, gae)
            
        return torch.stack(adv)
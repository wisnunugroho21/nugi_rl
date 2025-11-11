import torch
from torch import Tensor

from nugi_rl.distribution.base import Distribution
from nugi_rl.loss.ppo.base import Ppo


class PpoClip(Ppo):
    def __init__(self, distribution: Distribution, policy_clip: float = 0.2):
        super().__init__()

        self.policy_clip = policy_clip
        self.distribution = distribution

    def forward(
        self,
        action_datas: Tensor,
        old_action_datas: Tensor,
        actions: Tensor,
        advantages: Tensor,
    ) -> Tensor:
        logprobs = self.distribution.logprob(action_datas, actions)
        old_logprobs = (
            self.distribution.logprob(old_action_datas, actions) + 1e-3
        ).detach()

        ratios = (logprobs - old_logprobs).exp()
        surr1 = ratios * advantages
        surr2 = ratios.clamp(1 - self.policy_clip, 1 + self.policy_clip) * advantages
        loss = torch.min(surr1, surr2).mean()

        return -1 * loss

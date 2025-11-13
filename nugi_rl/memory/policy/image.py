from torch import Tensor

from nugi_rl.memory.policy.standard import StandardPolicyMemory
from nugi_rl.utilities.augmentation.base import Augmentation


class ImagePolicyMemory(StandardPolicyMemory):
    def __init__(self, trans: Augmentation, capacity: int = 100000):
        super().__init__(capacity=capacity)

        self.trans = trans

    def __getitem__(
        self, idx: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.trans(self.states[idx]),
            self.actions[idx],
            self.rewards[idx].unsqueeze(-1),
            self.dones[idx].unsqueeze(-1),
            self.trans(self.next_states[idx]),
            self.logprobs[idx],
        )

import torch
import numpy as np
from nugi_rl.environment.wrapper.gym_wrapper import GymWrapper
from torch import device, Tensor

class DiscretizationGymWrapper(GymWrapper):
    def __init__(self, env, agent_device: device, bins: int = 15) -> None:
        super().__init__(env, agent_device)

        self.act = torch.arange(bins, device = agent_device)
        self.act = (2 * self.act) / (bins - 1) - 1

    def step(self, action: Tensor) -> Tensor:
        action = self.act[action]
        return action
from typing import List, Union

import torch
from torch import Tensor, device
from torch.nn import Module
from torch.optim import Optimizer

from nugi_rl.agent.ppo import AgentPPO
from nugi_rl.distribution.base import Distribution
from nugi_rl.loss.entropy import EntropyLoss
from nugi_rl.loss.ppo.base import Ppo
from nugi_rl.loss.value import ValueLoss
from nugi_rl.memory.policy.image import ImagePolicyMemory
from nugi_rl.policy_function.advantage_function.gae import (
    GeneralizedAdvantageEstimation,
)
from nugi_rl.utilities.augmentation.base import Augmentation


class AgentImagePPO(AgentPPO):
    def __init__(
        self,
        policy: Module,
        value: Module,
        gae: GeneralizedAdvantageEstimation,
        distribution: Distribution,
        policy_loss: Ppo,
        value_loss: ValueLoss,
        entropy_loss: EntropyLoss,
        memory: ImagePolicyMemory,
        optimizer: Optimizer,
        trans: Augmentation,
        ppo_epochs: int = 10,
        is_training_mode: bool = True,
        batch_size: int = 32,
        folder: str = "model",
        device: device = torch.device("cuda:0"),
        policy_old: Module | None = None,
        value_old: Module | None = None,
        dont_unsqueeze=False,
    ):
        super().__init__(
            policy,
            value,
            gae,
            distribution,
            policy_loss,
            value_loss,
            entropy_loss,
            memory,
            optimizer,
            ppo_epochs,
            is_training_mode,
            batch_size,
            folder,
            device,
            policy_old,
            value_old,
            dont_unsqueeze,
        )
        self.trans = trans

    def act(self, state: Tensor) -> Tensor:
        with torch.inference_mode():
            state = state if self.dont_unsqueeze else state.unsqueeze(0)
            action_datas = self.policy(state)

            if self.is_training_mode:
                action = self.distribution.sample(action_datas)
            else:
                action = self.distribution.deterministic(action_datas)

            action = action.squeeze(0).detach()

        return action

    def logprobs(self, state, action: Tensor) -> Tensor:
        with torch.inference_mode():
            state = state if self.dont_unsqueeze else state.unsqueeze(0)
            action = action.unsqueeze(0)

            action_datas = self.policy(state)

            logprobs = self.distribution.logprob(action_datas, action)
            logprobs = logprobs.squeeze(0).detach()

        return logprobs

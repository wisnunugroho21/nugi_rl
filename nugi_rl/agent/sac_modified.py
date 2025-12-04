from copy import deepcopy

import torch
from torch import Tensor, device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler

from nugi_rl.agent.sac import AgentSac
from nugi_rl.distribution.base import Distribution
from nugi_rl.helpers.pytorch_utils import copy_parameters
from nugi_rl.loss.sac.policy_loss import PolicyLoss
from nugi_rl.loss.sac.q_loss import QLoss
from nugi_rl.memory.policy.base import PolicyMemory


class AgentSACModified(AgentSac):
    def __init__(
        self,
        soft_q1: Module,
        soft_q2: Module,
        policy: Module,
        distribution: Distribution,
        q_loss: QLoss,
        policy_loss: PolicyLoss,
        memory: PolicyMemory,
        soft_q_optimizer: Optimizer,
        policy_optimizer: Optimizer,
        is_training_mode: bool = True,
        batch_size: int = 32,
        epochs: int = 1,
        soft_tau: float = 0.95,
        policy_update_delay: int = 2,
        folder: str = "model",
        device: device = torch.device("cuda:0"),
        target_q1: Module | None = None,
        target_q2: Module | None = None,
        target_policy: Module | None = None,
        dont_unsqueeze=False,
    ) -> None:
        super().__init__(
            soft_q1,
            soft_q2,
            policy,
            distribution,
            q_loss,
            policy_loss,
            memory,
            soft_q_optimizer,
            policy_optimizer,
            is_training_mode,
            batch_size,
            epochs,
            soft_tau,
            folder,
            device,
            target_q1,
            target_q2,
            dont_unsqueeze,
        )

        self.policy_update_delay = policy_update_delay

        if target_policy is None:
            self.target_policy = deepcopy(self.policy)
        else:
            self.target_policy = target_policy

    def _update_step_q(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        next_states: Tensor,
    ) -> None:
        self.soft_q_optimizer.zero_grad()

        next_action_datas = self.target_policy(next_states)
        next_actions = self.distribution.sample(next_action_datas)

        predicted_q1 = self.soft_q1(states, actions)
        predicted_q2 = self.soft_q2(states, actions)

        target_next_q1 = self.target_q1(next_states, next_actions)
        target_next_q2 = self.target_q2(next_states, next_actions)

        loss = self.qLoss(
            predicted_q1,
            predicted_q2,
            target_next_q1,
            target_next_q2,
            next_action_datas,
            next_actions,
            rewards,
            dones,
            torch.Tensor([0.2]),
        )

        loss.backward()
        self.soft_q_optimizer.step()

    def update(self, config: str = "") -> None:
        dataloader = DataLoader(
            self.memory,
            self.batch_size,
            sampler=RandomSampler(self.memory, False),
        )

        i = 0
        if self.q_update == self.policy_update_delay:
            for states, actions, rewards, dones, next_states, _ in dataloader:
                self._update_step_q(states, actions, rewards, dones, next_states)
                self._update_step_policy(states)

                self.target_q1 = copy_parameters(
                    self.soft_q1, self.target_q1, self.soft_tau
                )
                self.target_q2 = copy_parameters(
                    self.soft_q2, self.target_q2, self.soft_tau
                )

                self.target_policy = copy_parameters(
                    self.policy, self.target_policy, self.soft_tau
                )

                if i == self.epochs:
                    break

            self.q_update = 0
        else:
            for states, actions, rewards, dones, next_states, _ in dataloader:
                self._update_step_q(states, actions, rewards, dones, next_states)

                self.target_q1 = copy_parameters(
                    self.soft_q1, self.target_q1, self.soft_tau
                )
                self.target_q2 = copy_parameters(
                    self.soft_q2, self.target_q2, self.soft_tau
                )

                if i == self.epochs:
                    break

            self.q_update += 1

    def load_weights(self) -> None:
        model_checkpoint = torch.load(
            self.folder + "/sac.tar", map_location=self.device
        )

        self.policy.load_state_dict(model_checkpoint["policy_state_dict"])
        self.soft_q1.load_state_dict(model_checkpoint["soft_q1_state_dict"])
        self.soft_q2.load_state_dict(model_checkpoint["soft_q2_state_dict"])
        self.target_policy.load_state_dict(model_checkpoint["target_policy_state_dict"])
        self.target_q1.load_state_dict(model_checkpoint["target_q1_state_dict"])
        self.target_q2.load_state_dict(model_checkpoint["target_q2_state_dict"])
        self.policy_optimizer.load_state_dict(
            model_checkpoint["policy_optimizer_state_dict"]
        )
        self.soft_q_optimizer.load_state_dict(
            model_checkpoint["soft_q_optimizer_state_dict"]
        )

    def save_weights(self) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "soft_q1_state_dict": self.soft_q1.state_dict(),
                "soft_q2_state_dict": self.soft_q2.state_dict(),
                "target_policy_state_dict": self.target_policy.state_dict(),
                "target_q1_state_dict": self.target_q1.state_dict(),
                "target_q2_state_dict": self.target_q2.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "soft_q_optimizer_state_dict": self.soft_q_optimizer.state_dict(),
            },
            self.folder + "/sac.tar",
        )

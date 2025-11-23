from copy import deepcopy

import torch
from torch import Tensor, device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, SubsetRandomSampler

from nugi_rl.agent.base import Agent
from nugi_rl.distribution.base import Distribution
from nugi_rl.helpers.pytorch_utils import copy_parameters
from nugi_rl.loss.sac.policy_loss import PolicyLoss
from nugi_rl.loss.sac.q_loss import QLoss
from nugi_rl.memory.policy.base import PolicyMemory


class AgentSac(Agent):
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
        folder: str = "model",
        device: device = torch.device("cuda:0"),
        target_q1: Module | None = None,
        target_q2: Module | None = None,
        dont_unsqueeze=False,
    ) -> None:
        self.dont_unsqueeze = dont_unsqueeze
        self.batch_size = batch_size
        self.is_training_mode = is_training_mode
        self.folder = folder
        self.epochs = epochs
        self.soft_tau = soft_tau

        self.policy = policy
        self.soft_q1 = soft_q1
        self.soft_q2 = soft_q2

        self.distribution = distribution
        self.memory = memory

        self.qLoss = q_loss
        self.policyLoss = policy_loss

        self.device = device
        self.q_update = 1

        self.soft_q_optimizer = soft_q_optimizer
        self.policy_optimizer = policy_optimizer

        if target_q1 is None:
            self.target_q1 = deepcopy(self.soft_q1)
        else:
            self.target_q1 = target_q1

        if target_q2 is None:
            self.target_q2 = deepcopy(self.soft_q2)
        else:
            self.target_q2 = target_q2

    def _update_step_policy(self, states: Tensor) -> None:
        self.policy_optimizer.zero_grad()

        action_datas = self.policy(states)
        actions = self.distribution.sample(action_datas)

        q_value1 = self.soft_q1(states, actions)
        q_value2 = self.soft_q2(states, actions)

        loss = self.policyLoss(
            action_datas, actions, q_value1, q_value2, torch.Tensor([0.2])
        )

        loss.backward()
        self.policy_optimizer.step()

    def _update_step_q(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        next_states: Tensor,
    ) -> None:
        self.soft_q_optimizer.zero_grad()

        next_action_datas = self.policy(next_states)
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

    def act(self, state: Tensor) -> Tensor:
        with torch.inference_mode():
            state = state if self.dont_unsqueeze else state.unsqueeze(0)
            action_datas = self.policy(state)

            if self.is_training_mode:
                action = self.distribution.sample(action_datas)
            else:
                action = self.distribution.deterministic(action_datas)

            action = action.squeeze(0)

        return action

    def logprob(self, state: Tensor, action: Tensor) -> Tensor:
        with torch.inference_mode():
            state = state if self.dont_unsqueeze else state.unsqueeze(0)
            action = action if self.dont_unsqueeze else action.unsqueeze(0)

            action_datas = self.policy(state)

            logprobs = self.distribution.logprob(action_datas, action)
            logprobs = logprobs.squeeze(0)

        return logprobs

    def update(self, config: str = "") -> None:
        for _ in range(self.epochs):
            indices = torch.randperm(len(self.memory))[: self.batch_size - 1]
            indices = torch.concat(
                (indices, torch.tensor([len(self.memory) - 1])), dim=0
            )

            dataloader = DataLoader(
                self.memory, self.batch_size, sampler=SubsetRandomSampler(indices)
            )
            for states, actions, rewards, dones, next_states, _ in dataloader:
                self._update_step_q(states, actions, rewards, dones, next_states)
                self._update_step_policy(states)

                self.target_q1 = copy_parameters(
                    self.soft_q1, self.target_q1, self.soft_tau
                )
                self.target_q2 = copy_parameters(
                    self.soft_q2, self.target_q2, self.soft_tau
                )

    def save_obs(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        next_state: Tensor,
        logprob: Tensor,
    ) -> None:
        self.memory.save(state, action, reward, done, next_state, logprob)

    def save_all(
        self,
        states: list[Tensor],
        actions: list[Tensor],
        rewards: list[Tensor],
        dones: list[Tensor],
        next_states: list[Tensor],
        logprobs: list[Tensor],
    ) -> None:
        self.memory.save_all(states, actions, rewards, dones, next_states, logprobs)

    def get_obs(
        self, start_position: int = 0, end_position: int | None = None
    ) -> tuple[
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
    ]:
        return self.memory.get(start_position, end_position)

    def clear_obs(
        self, start_position: int = 0, end_position: int | None = None
    ) -> None:
        self.memory.clear(start_position, end_position)

    def load_weights(self) -> None:
        model_checkpoint = torch.load(
            self.folder + "/sac.tar", map_location=self.device
        )

        self.policy.load_state_dict(model_checkpoint["policy_state_dict"])
        self.soft_q1.load_state_dict(model_checkpoint["soft_q1_state_dict"])
        self.soft_q2.load_state_dict(model_checkpoint["soft_q2_state_dict"])
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
                "target_q1_state_dict": self.target_q1.state_dict(),
                "target_q2_state_dict": self.target_q2.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "soft_q_optimizer_state_dict": self.soft_q_optimizer.state_dict(),
            },
            self.folder + "/sac.tar",
        )

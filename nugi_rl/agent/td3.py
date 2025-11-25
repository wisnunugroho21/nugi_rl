from copy import deepcopy

import torch
from torch import Tensor, device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler

from nugi_rl.agent.base import Agent
from nugi_rl.helpers.pytorch_utils import copy_parameters
from nugi_rl.loss.td3.policy_loss import PolicyLoss
from nugi_rl.loss.td3.q_loss import QLoss
from nugi_rl.memory.policy.base import PolicyMemory


class AgentTd3(Agent):
    def __init__(
        self,
        soft_q1: Module,
        soft_q2: Module,
        policy: Module,
        q_loss: QLoss,
        policy_loss: PolicyLoss,
        memory: PolicyMemory,
        soft_q_optimizer: Optimizer,
        policy_optimizer: Optimizer,
        is_training_mode: bool = True,
        batch_size: int = 32,
        epochs: int = 1,
        soft_tau: float = 0.95,
        action_noise_std: float = 0.1,
        target_action_noise: tuple[float, float] = (-0.2, 0.2),
        action_range: tuple[float, float] = (-1.0, 1.0),
        policy_update_delay: int = 2,
        folder: str = "model",
        device: device = torch.device("cuda:0"),
        target_policy: Module | None = None,
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
        self.action_noise_std = action_noise_std
        self.target_action_noise = target_action_noise
        self.action_range = action_range
        self.policy_update_delay = policy_update_delay

        self.policy = policy
        self.soft_q1 = soft_q1
        self.soft_q2 = soft_q2

        self.memory = memory

        self.qLoss = q_loss
        self.policyLoss = policy_loss

        self.device = device
        self.q_update = 0

        self.soft_q_optimizer = soft_q_optimizer
        self.policy_optimizer = policy_optimizer

        if target_policy is None:
            self.target_policy = deepcopy(self.policy)
        else:
            self.target_policy = target_policy

        if target_q1 is None:
            self.target_q1 = deepcopy(self.soft_q1)
        else:
            self.target_q1 = target_q1

        if target_q2 is None:
            self.target_q2 = deepcopy(self.soft_q2)
        else:
            self.target_q2 = target_q2

    def _update_step_q(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        next_states: Tensor,
    ) -> None:
        self.soft_q_optimizer.zero_grad()

        predicted_q1: Tensor = self.soft_q1(states, actions)
        predicted_q2: Tensor = self.soft_q2(states, actions)

        action_noise = torch.normal(torch.tensor([0.0]), torch.tensor([1.0]))
        action_noise = action_noise.clamp(
            self.target_action_noise[0], self.target_action_noise[1]
        )

        target_next_actions: Tensor = self.target_policy(next_states)
        target_next_actions = (target_next_actions + action_noise).clamp(
            self.action_range[0], self.action_range[1]
        )

        target_next_q1: Tensor = self.target_q1(next_states, target_next_actions)
        target_next_q2: Tensor = self.target_q2(next_states, target_next_actions)

        loss: Tensor = self.qLoss(
            predicted_q1, predicted_q2, target_next_q1, target_next_q2, rewards, dones
        )

        loss.backward()
        self.soft_q_optimizer.step()

    def _update_step_policy(self, states) -> None:
        self.policy_optimizer.zero_grad()

        actions: Tensor = self.policy(states)

        q_value1: Tensor = self.soft_q1(states, actions)
        q_value2: Tensor = self.soft_q2(states, actions)

        loss: Tensor = self.policyLoss(q_value1, q_value2)

        loss.backward()
        self.policy_optimizer.step()

    def act(self, state: Tensor) -> Tensor:
        with torch.inference_mode():
            action_noise = torch.normal(
                torch.tensor([0.0]), torch.tensor([self.action_noise_std])
            )

            state = state if self.dont_unsqueeze else state.unsqueeze(0)
            action = self.policy(state).squeeze(0)
            action = (action + action_noise).clamp(
                self.action_range[0], self.action_range[1]
            )
        return action

    def logprob(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.tensor([0], device=self.device)

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

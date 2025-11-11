import torch
from torch import Tensor, device
from torch.nn import Module
from torch.optim import Optimizer

from nugi_rl.agent.sac import AgentSac
from nugi_rl.distribution.base import Distribution
from nugi_rl.loss.sac.policy_loss import PolicyLoss
from nugi_rl.loss.sac.q_loss import QLoss
from nugi_rl.loss.sac.temperature_loss import TemperatureLoss
from nugi_rl.memory.policy.base import PolicyMemory


class AgentSACtemperature(AgentSac):
    def __init__(
        self,
        soft_q1: Module,
        soft_q2: Module,
        policy: Module,
        distribution: Distribution,
        q_loss: QLoss,
        policy_loss: PolicyLoss,
        temperature_loss: TemperatureLoss,
        memory: PolicyMemory,
        soft_q_optimizer: Optimizer,
        policy_optimizer: Optimizer,
        is_training_mode: bool = True,
        batch_size: int = 32,
        epochs: int = 1,
        soft_tau: float = 0.95,
        folder: str = "model",
        device: device = torch.device("cuda:0"),
        target_policy: Module | None = None,
        target_q1: Module | None = None,
        target_q2: Module | None = None,
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
            target_policy,
            target_q1,
            target_q2,
            dont_unsqueeze,
        )

        self.temperature_loss = temperature_loss

    def _update_step_policy(self, states: Tensor) -> None:
        self.policy_optimizer.zero_grad()

        action_datas, alpha = self.policy(states)
        actions = self.distribution.sample(action_datas)

        target_action_datas, _ = self.target_policy(states)
        target_actions = self.distribution.sample(*target_action_datas)

        q_value1 = self.soft_q1(states, actions)
        q_value2 = self.soft_q2(states, actions)

        loss = self.policyLoss(
            action_datas, actions, q_value1, q_value2, alpha
        ) + self.temperature_loss(target_action_datas, target_actions, alpha)

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

        _, alpha = self.policy(states)

        next_action_datas, _ = self.target_policy(next_states)
        next_actions = self.distribution.sample(*next_action_datas)

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
            alpha,
        )

        loss.backward()
        self.soft_q_optimizer.step()

    def act(self, state: Tensor) -> Tensor:
        with torch.inference_mode():
            state = state if self.dont_unsqueeze else state.unsqueeze(0)
            action_datas, _ = self.policy(state)

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

            action_datas, _ = self.policy(state)

            logprobs = self.distribution.logprob(action_datas, action)
            logprobs = logprobs.squeeze(0)

        return logprobs

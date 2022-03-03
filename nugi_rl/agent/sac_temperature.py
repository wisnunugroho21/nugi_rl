import torch
from torch.nn import Module
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import Optimizer
from torch import device, Tensor

from copy import deepcopy
from typing import List, Union
from nugi_rl.agent.sac import AgentSac

from nugi_rl.distribution.base import Distribution
from nugi_rl.agent.base import Agent
from nugi_rl.loss.sac_temperature.policy_loss import PolicyLoss
from nugi_rl.loss.sac_temperature.q_loss import QLoss
from nugi_rl.loss.sac_temperature.temperature_loss import TemperatureLoss
from nugi_rl.memory.policy.base import PolicyMemory
from nugi_rl.helpers.pytorch_utils import copy_parameters

class AgentSACtemperature(AgentSac):
    def __init__(self, soft_q1: Module, soft_q2: Module, policy: Module, distribution: Distribution, q_loss: QLoss, policy_loss: PolicyLoss, temperature_loss: TemperatureLoss,
        memory: PolicyMemory, soft_q_optimizer: Optimizer, policy_optimizer: Optimizer, is_training_mode: bool = True, batch_size: int = 32, epochs: int = 1, soft_tau: float = 0.95, 
        folder: str = 'model', device: device = torch.device('cuda:0'), target_policy: Module = None, target_q1: Module = None, target_q2: Module = None, dont_unsqueeze=False) -> None:

        super().__init__(soft_q1, soft_q2, policy, distribution, q_loss, policy_loss, memory, soft_q_optimizer, policy_optimizer, is_training_mode, 
            batch_size, epochs, soft_tau, folder, device, target_policy, target_q1, target_q2, dont_unsqueeze)

        self.temperature_loss = temperature_loss

    def _update_step_policy(self, states: Tensor) -> None:
        self.policy_optimizer.zero_grad()

        action_datas, alpha = self.policy(states)
        actions             = self.distribution.sample(*action_datas)
        alpha               = alpha.detach()

        q_value1        = self.soft_q1(states, actions)
        q_value2        = self.soft_q2(states, actions)

        loss = self.policyLoss(action_datas, actions, q_value1, q_value2, alpha)

        loss.backward()
        self.policy_optimizer.step()

    def _update_step_q(self, states: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor, next_states: Tensor) -> Tensor:
        self.soft_q_optimizer.zero_grad()

        _, alpha                = self.policy(states)

        next_action_datas, _    = self.target_policy(next_states)
        next_actions            = self.distribution.sample(*next_action_datas)

        predicted_q1            = self.soft_q1(states, actions)
        predicted_q2            = self.soft_q2(states, actions)

        target_next_q1          = self.target_q1(next_states, next_actions)
        target_next_q2          = self.target_q2(next_states, next_actions)

        loss  = self.qLoss(predicted_q1, predicted_q2, target_next_q1, target_next_q2, next_action_datas, next_actions, rewards, dones, alpha)

        loss.backward()
        self.soft_q_optimizer.step()

    def _update_step_temperature(self, states: Tensor) -> None:
        self.policy_optimizer.zero_grad()

        with torch.no_grad():
            action_datas, alpha = self.policy(states)
            actions             = self.distribution.sample(*action_datas)

        loss = self.temperature_loss(action_datas, actions, alpha)

        loss.backward()
        self.policy_optimizer.step()    

    def act(self, state: Union[Tensor, List[Tensor]]) -> Tensor:
        with torch.inference_mode():
            if isinstance(state, list):
                for i in range(len(state)):
                    state[i] = state[i] if self.dont_unsqueeze else state[i].unsqueeze(0)
            else:
                state = state if self.dont_unsqueeze else state.unsqueeze(0)

            action_datas, _ = self.policy(state)
            
            if self.is_training_mode:
                action = self.distribution.sample(*action_datas)
            else:
                action = self.distribution.deterministic(action_datas)

            action = action.squeeze(0)
              
        return action

    def logprob(self, state: Union[Tensor, List[Tensor]], action: Tensor) -> Tensor:
        with torch.inference_mode():
            if isinstance(state, list):
                for i in range(len(state)):
                    state[i] = state[i] if self.dont_unsqueeze else state[i].unsqueeze(0)
            else:
                state = state if self.dont_unsqueeze else state.unsqueeze(0)

            action          = action if self.dont_unsqueeze else action.unsqueeze(0)
            action_datas, _ = self.policy(state)

            logprobs        = self.distribution.logprob(*action_datas, action)
            logprobs        = logprobs.squeeze(0)

        return logprobs

    def update(self) -> None:
        for _ in range(self.epochs):
            indices     = torch.randperm(len(self.memory))[:self.batch_size - 1]
            indices     = torch.concat((indices, torch.tensor([len(self.memory) - 1])), dim = 0)

            dataloader  = DataLoader(self.memory, self.batch_size, sampler = SubsetRandomSampler(indices))                
            for states, actions, rewards, dones, next_states, _ in dataloader:                
                self._update_step_q(states, actions, rewards, dones, next_states)
                self._update_step_policy(states)
                self._update_step_temperature(states)

                self.target_policy  = copy_parameters(self.policy, self.target_policy, self.soft_tau)
                self.target_q1      = copy_parameters(self.soft_q1, self.target_q1, self.soft_tau)
                self.target_q2      = copy_parameters(self.soft_q2, self.target_q2, self.soft_tau)

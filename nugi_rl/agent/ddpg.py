import torch
from torch.nn import Module
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import Optimizer
from torch import device, Tensor

from copy import deepcopy
from typing import List, Union

from nugi_rl.agent.base import Agent
from nugi_rl.loss.ddpg.policy_loss import PolicyLoss
from nugi_rl.loss.ddpg.q_loss import QLoss
from nugi_rl.memory.policy.base import PolicyMemory
from nugi_rl.helpers.pytorch_utils import copy_parameters

class AgentDdpg(Agent):
    def __init__(self, soft_q: Module, policy: Module, q_loss: QLoss, policy_loss: PolicyLoss, memory: PolicyMemory, 
        soft_q_optimizer: Optimizer, policy_optimizer: Optimizer, is_training_mode: bool = True, batch_size: int = 32, 
        epochs: int = 1, soft_tau: float = 0.95, folder: str = 'model', device: device = torch.device('cuda:0'), 
        target_policy: Module = None, target_q: Module = None, dont_unsqueeze = False) -> None:

        self.dont_unsqueeze     = dont_unsqueeze
        self.batch_size         = batch_size
        self.is_training_mode   = is_training_mode
        self.folder             = folder
        self.epochs             = epochs
        self.soft_tau           = soft_tau

        self.policy             = policy
        self.soft_q             = soft_q
        self.target_policy      = target_policy
        self.target_q           = target_q

        self.memory             = memory
        
        self.qLoss              = q_loss
        self.policyLoss         = policy_loss

        self.device             = device
        self.q_update           = 1
        
        self.soft_q_optimizer   = soft_q_optimizer
        self.policy_optimizer   = policy_optimizer

        if self.target_policy is None:
            self.target_policy = deepcopy(self.policy)

        if self.target_q is None:
            self.target_q = deepcopy(self.soft_q)

    def _update_step_policy(self, states: Tensor) -> None:
        self.policy_optimizer.zero_grad()

        actions = self.policy(states)
        q_value = self.soft_q(states, actions)

        loss = self.policyLoss(q_value)

        loss.backward()
        self.policy_optimizer.step()

    def _update_step_q(self, states: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor, next_states: Tensor) -> Tensor:
        self.soft_q_optimizer.zero_grad()

        next_actions    = self.target_policy(next_states)
        predicted_q     = self.soft_q(states, actions)
        target_next_q   = self.target_q(next_states, next_actions)

        loss  = self.qLoss(predicted_q, target_next_q, rewards, dones)

        loss.backward()
        self.soft_q_optimizer.step() 

    def act(self, state: Union[Tensor, List[Tensor]]) -> Tensor:
        with torch.inference_mode():
            if isinstance(state, list):
                state = [s if self.dont_unsqueeze else s.unsqueeze(0) for s in state]
            else:
                state = state if self.dont_unsqueeze else state.unsqueeze(0)

            action  = self.policy(state).squeeze(0)
              
        return action

    def logprob(self, state: Union[Tensor, List[Tensor]], action: Tensor) -> Tensor:
        return torch.tensor([0], device = self.device)

    def save_obs(self, state: Union[Tensor, List[Tensor]], action: Tensor, reward: Tensor, done: Tensor, next_state: Union[Tensor, List[Tensor]], logprob: Tensor) -> None:
        self.memory.save(state, action, reward, done, next_state, logprob)

    def save_all(self, states: Union[Tensor, List[Tensor]], actions: Tensor, rewards: Tensor, dones: Tensor, next_states: Union[Tensor, List[Tensor]], logprobs: Tensor) -> None:
        self.memory.save_all(states, actions, rewards, dones, next_states, logprobs)
        
    def update(self) -> None:
        for _ in range(self.epochs):
            indices     = torch.randperm(len(self.memory))[:self.batch_size - 1]
            indices     = torch.concat((indices, torch.tensor([len(self.memory) - 1])), dim = 0)

            dataloader  = DataLoader(self.memory, self.batch_size, sampler = SubsetRandomSampler(indices))                
            for states, actions, rewards, dones, next_states, _ in dataloader:                
                self._update_step_q(states.to(self.device), actions.to(self.device), rewards.to(self.device), dones.to(self.device), next_states.to(self.device))
                self._update_step_policy(states.to(self.device))

                self.target_policy  = copy_parameters(self.policy, self.target_policy, self.soft_tau)
                self.target_q = copy_parameters(self.soft_q, self.target_q, self.soft_tau)

    def get_obs(self, start_position: int = None, end_position: int = None) -> tuple:
        return self.memory.get(start_position, end_position)

    def clear_obs(self, start_position: int = 0, end_position: int = None) -> None:
        self.memory.clear(start_position, end_position)

    def load_weights(self) -> None:
        model_checkpoint = torch.load(self.folder + '/ddpg.tar', map_location = self.device)
        
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])
        self.soft_q.load_state_dict(model_checkpoint['soft_q_state_dict'])
        self.target_q.load_state_dict(model_checkpoint['target_q_state_dict'])
        self.policy_optimizer.load_state_dict(model_checkpoint['policy_optimizer_state_dict'])
        self.soft_q_optimizer.load_state_dict(model_checkpoint['soft_q_optimizer_state_dict'])

    def save_weights(self) -> None:
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'soft_q_state_dict': self.soft_q.state_dict(),
            'target_q_state_dict': self.target_q.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'soft_q_optimizer_state_dict': self.soft_q_optimizer.state_dict(),
        }, self.folder + '/ddpg.tar')       
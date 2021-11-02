import torch
from torch.nn import Module
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import Optimizer
from torch import device

from copy import deepcopy

from nugi_rl.agent.base import Agent
from nugi_rl.loss.td3.policy_loss import PolicyLoss
from nugi_rl.loss.td3.q_loss import QLoss
from nugi_rl.loss.cql_regularizer import CqlRegularizer
from nugi_rl.memory.base import Memory
from nugi_rl.helpers.pytorch_utils import copy_parameters

class AgentSac(Agent):
    def __init__(self, soft_q1: Module, soft_q2: Module, policy: Module, q_loss: QLoss, policy_loss: PolicyLoss, cql_reg_loss: CqlRegularizer, 
        memory: Memory, soft_q_optimizer: Optimizer, policy_optimizer: Optimizer, is_training_mode: bool = True, batch_size: int = 32, epochs: int = 1, 
        soft_tau: float = 0.95, folder: str = 'model', device: device = torch.device('cuda:0'), target_q1: Module = None, target_q2: Module = None):

        self.batch_size         = batch_size
        self.is_training_mode   = is_training_mode
        self.folder             = folder
        self.epochs             = epochs
        self.soft_tau           = soft_tau

        self.policy             = policy
        self.soft_q1            = soft_q1
        self.soft_q2            = soft_q2

        self.target_q1          = target_q1
        self.target_q2          = target_q2

        self.memory             = memory
        
        self.qLoss              = q_loss
        self.policyLoss         = policy_loss
        self.cqlRegLoss         = cql_reg_loss

        self.device             = device
        self.q_update           = 1
        
        self.soft_q_optimizer   = soft_q_optimizer
        self.policy_optimizer   = policy_optimizer

        if self.target_q1 is None:
            self.target_q1 = deepcopy(self.soft_q1)

        if self.target_q2 is None:
            self.target_q2 = deepcopy(self.soft_q2)

    def _update_step_q(self, states, actions, rewards, dones, next_states):
        self.soft_q_optimizer.zero_grad()        

        predicted_q1        = self.soft_q1(states, actions)
        predicted_q2        = self.soft_q2(states, actions)

        predicted_actions   = self.policy(states, True)
        naive_q1_value      = self.soft_q1(states, predicted_actions)
        naive_q2_value      = self.soft_q2(states, predicted_actions)

        next_actions        = self.policy(next_states, True)
        target_next_q1      = self.target_q1(next_states, next_actions, True)
        target_next_q2      = self.target_q2(next_states, next_actions, True)

        loss  = self.qLoss.compute_loss(predicted_q1, predicted_q2, target_next_q1, target_next_q2, rewards, dones) + \
            self.cqlRegLoss.compute_loss(predicted_q1, predicted_q2, naive_q1_value, naive_q2_value)

        loss.backward()
        self.soft_q_optimizer.step()   

    def _update_step_policy(self, states):
        self.policy_optimizer.zero_grad()

        actions     = self.policy(states)

        q_value1    = self.soft_q1(states, actions)
        q_value2    = self.soft_q2(states, actions)

        loss = self.policyLoss.compute_loss(q_value1, q_value2)

        loss.backward()
        self.policy_optimizer.step()    

    def act(self, state: list) -> list:
        with torch.inference_mode():
            state   = torch.tensor(state).float().to(self.device).unsqueeze(0)
            action  = self.policy(state)
            action  = action.squeeze(0).detach().tolist()
              
        return action

    def logprob(self, state: list, action: list) -> list:
        raise NotImplementedError('TD3 is deterministic')

    def save_obs(self, state: list, action: list, reward: float, done: bool, next_state: list) -> None:
        self.memory.save(state, action, reward, done, next_state)

    def save_memory(self, memory: Memory) -> None:
        states, actions, rewards, dones, next_states = memory.get()
        self.memory.save_all(states, actions, rewards, dones, next_states)
        
    def update(self) -> None:
        for _ in range(self.epochs):
            indices     = torch.randperm(len(self.memory))[:self.batch_size]
            indices[-1] = torch.IntTensor([len(self.memory) - 1])

            dataloader  = DataLoader(self.memory, self.batch_size, sampler = SubsetRandomSampler(indices))                
            for states, actions, rewards, dones, next_states in dataloader:                
                self._update_step_q(states.to(self.device), actions.to(self.device), rewards.to(self.device), dones.to(self.device), next_states.to(self.device))
                self._update_step_policy(states.to(self.device))

                self.target_q1 = copy_parameters(self.soft_q1, self.target_q1, self.soft_tau)
                self.target_q2 = copy_parameters(self.soft_q2, self.target_q2, self.soft_tau)

    def get_obs(self, start_position: int = None, end_position: int = None) -> tuple:
        return self.memory.get(start_position, end_position)

    def load_weights(self) -> None:
        model_checkpoint = torch.load(self.folder + '/sac.tar', map_location = self.device)
        
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])
        self.soft_q1.load_state_dict(model_checkpoint['soft_q1_state_dict'])
        self.soft_q2.load_state_dict(model_checkpoint['soft_q2_state_dict'])
        self.target_q1.load_state_dict(model_checkpoint['target_q1_state_dict'])
        self.target_q2.load_state_dict(model_checkpoint['target_q2_state_dict'])
        self.policy_optimizer.load_state_dict(model_checkpoint['policy_optimizer_state_dict'])
        self.soft_q_optimizer.load_state_dict(model_checkpoint['soft_q_optimizer_state_dict'])

    def save_weights(self) -> None:
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'soft_q1_state_dict': self.soft_q1.state_dict(),
            'soft_q2_state_dict': self.soft_q2.state_dict(),
            'target_q1_state_dict': self.target_q1.state_dict(),
            'target_q2_state_dict': self.target_q2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'soft_q_optimizer_state_dict': self.soft_q_optimizer.state_dict(),
        }, self.folder + '/sac.tar')     
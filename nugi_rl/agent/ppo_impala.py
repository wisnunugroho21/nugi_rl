import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import device

from copy import deepcopy
from typing import List, Union

from nugi_rl.distribution.base import Distribution
from nugi_rl.agent.base import Agent
from nugi_rl.loss.ppo.base import Ppo
from nugi_rl.loss.value import ValueLoss
from nugi_rl.loss.entropy import EntropyLoss
from nugi_rl.memory.policy.base import PolicyMemory
from nugi_rl.policy_function.advantage_function.vtrace import VtraceAdvantageEstimation

class AgentImpalaPPO(Agent):  
    def __init__(self, policy: Module, value: Module, gae: VtraceAdvantageEstimation, distribution: Distribution, 
        policy_loss: Ppo, value_loss: ValueLoss, entropy_loss: EntropyLoss, memory: PolicyMemory, optimizer: Optimizer, 
        ppo_epochs: int = 10, is_training_mode: bool = True, batch_size: int = 32, folder: str = 'model', 
        device: device = torch.device('cuda:0'), policy_old: Module = None, value_old: Module = None, dont_unsqueeze = False) -> None:

        self.dont_unsqueeze     = dont_unsqueeze
        self.batch_size         = batch_size  
        self.ppo_epochs         = ppo_epochs
        self.is_training_mode   = is_training_mode
        self.folder             = folder

        self.policy             = policy
        self.policy_old         = policy_old

        self.value              = value
        self.value_old          = value_old

        self.distribution       = distribution
        self.memory             = memory
        self.gae                = gae
        
        self.policy_loss        = policy_loss
        self.value_loss         = value_loss
        self.entropy_loss       = entropy_loss

        self.optimizer          = optimizer
        self.device             = device

        if self.policy_old is None:
            self.policy_old = deepcopy(self.policy)

        if self.value_old is None:
            self.value_old  = deepcopy(self.value)

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()

    def _update_step(self, states: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor, next_states: Tensor, logprobs: Tensor) -> None:
        self.optimizer.zero_grad()

        action_datas        = self.policy(states)
        values              = self.value(states)

        cur_logprobs        = self.logprob(states, actions)

        old_action_datas    = self.policy_old(states)
        old_values          = self.value_old(states)
        next_values         = self.value(next_states)

        adv = self.gae(rewards, values, next_values, dones, logprobs, cur_logprobs).detach()

        loss = self.policy_loss(action_datas, old_action_datas, actions, adv) + \
            self.value_loss(values, adv, old_values) + \
            self.entropy_loss(action_datas)
        
        loss.backward()
        self.optimizer.step()

    def act(self, state: Union[Tensor, List[Tensor]]) -> Tensor:
        with torch.inference_mode():
            if isinstance(state, list):
                state = [s if self.dont_unsqueeze else s.unsqueeze(0) for s in state]
            else:
                state = state if self.dont_unsqueeze else state.unsqueeze(0)

            action_datas    = self.policy(state)
            
            if self.is_training_mode:
                action = self.distribution.sample(*action_datas)
            else:
                action = self.distribution.deterministic(*action_datas)

            action = action.squeeze(0)
              
        return action

    def logprob(self, state: Union[Tensor, List[Tensor]], action: Tensor) -> Tensor:
        with torch.inference_mode():
            if isinstance(state, list):
                state = [s if self.dont_unsqueeze else s.unsqueeze(0) for s in state]
            else:
                state = state if self.dont_unsqueeze else state.unsqueeze(0)

            action          = action if self.dont_unsqueeze else action.unsqueeze(0)
            action_datas    = self.policy(state)

            logprobs        = self.distribution.logprob(*action_datas, action)
            logprobs        = logprobs.squeeze(0)

        return logprobs

    def update(self) -> None:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

        for _ in range(self.ppo_epochs):
            dataloader = DataLoader(self.memory, self.batch_size, shuffle = False)
            for states, actions, rewards, dones, next_states, logprobs in dataloader:
               self._update_step(states, actions, rewards, dones, next_states, logprobs)

        self.memory.clear()

    def save_obs(self, state: Tensor, action: Tensor, reward: Tensor, done: Tensor, next_state: Tensor, logprob: Tensor) -> None:
        self.memory.save(state, action, reward, done, next_state, logprob)

    def save_all(self, states: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor, next_states: Tensor, logprobs: Tensor) -> None:
        self.memory.save_all(states, actions, rewards, dones, next_states, logprobs)    

    def get_obs(self, start_position: int = None, end_position: int = None) -> tuple:
        return self.memory.get(start_position, end_position)

    def clear_obs(self, start_position: int = 0, end_position: int = None) -> None:
        self.memory.clear(start_position, end_position)

    def load_weights(self) -> None:
        model_checkpoint = torch.load(self.folder + '/ppo.tar', map_location = self.device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(model_checkpoint['ppo_optimizer_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.value.train()

        else:
            self.policy.eval()
            self.value.eval()

    def save_weights(self) -> None:            
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'ppo_optimizer_state_dict': self.optimizer.state_dict(),
        }, self.folder + '/ppo.tar')
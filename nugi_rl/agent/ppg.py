import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import device, Tensor

from copy import deepcopy

from nugi_rl.distribution.base import Distribution
from nugi_rl.agent.base import Agent
from nugi_rl.loss.ppo.base import Ppo
from nugi_rl.loss.value import ValueLoss
from nugi_rl.loss.entropy import EntropyLoss
from nugi_rl.loss.ppg import PpgLoss
from nugi_rl.memory.policy.base import PolicyMemory
from nugi_rl.memory.ppg import PPGMemory
from nugi_rl.policy_function.advantage_function.gae import GeneralizedAdvantageEstimation

class AgentPPO(Agent):  
    def __init__(self, policy: Module, value: Module, gae: GeneralizedAdvantageEstimation, distribution: Distribution, 
        policy_loss: Ppo, value_loss: ValueLoss, entropy_loss: EntropyLoss, aux_loss: PpgLoss,
        policy_memory: PolicyMemory, aux_memory: PPGMemory, policy_optimizer: Optimizer, aux_optimizer: Optimizer, 
        policy_epochs: int = 10, aux_epochs: int = 10, n_aux_update: int = 5, is_training_mode: bool = True, 
        batch_size: int = 32, folder: str = 'model', device: device = torch.device('cuda:0'), 
        policy_old: Module = None, value_old: Module = None, dont_unsqueeze = False) -> None:

        self.dont_unsqueeze     = dont_unsqueeze
        self.batch_size         = batch_size  
        
        self.is_training_mode   = is_training_mode
        self.folder             = folder

        self.policy_epochs      = policy_epochs
        self.aux_epochs         = aux_epochs

        self.policy_memory      = policy_memory
        self.aux_memory         = aux_memory

        self.policy             = policy
        self.policy_old         = policy_old

        self.value              = value
        self.value_old          = value_old       
        
        self.policy_loss        = policy_loss
        self.value_loss         = value_loss
        self.entropy_loss       = entropy_loss
        self.aux_loss           = aux_loss

        self.policy_optimizer   = policy_optimizer
        self.aux_optimizer      = aux_optimizer

        self.n_aux_update       = n_aux_update
        self.distribution       = distribution
        self.gae                = gae
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

    def _update_policy_step(self, states: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor, next_states: Tensor) -> None:
        self.policy_optimizer.zero_grad()

        action_datas, _     = self.policy(states)
        values              = self.value(states)

        old_action_datas, _ = self.policy_old(states, True)
        old_values          = self.value_old(states, True)
        next_values         = self.value(next_states, True)

        adv = self.gae(rewards, values, next_values, dones).detach()

        loss = self.policy_loss(action_datas, old_action_datas, actions, adv) + \
            self.value_loss(values, adv, old_values) + \
            self.entropy_loss(action_datas)
        
        loss.backward()
        self.policy_optimizer.step()

    def _update_aux_step(self, states):
        self.aux_optimizer.zero_grad()

        action_datas, values    = self.policy(states)

        returns                 = self.value(states, True)
        old_action_datas, _     = self.policy_old(states, True)

        loss = self.aux_loss(action_datas, old_action_datas, values, returns)
        
        loss.backward()
        self.aux_optimizer.step()

    def _update_policy(self) -> None:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

        for _ in range(self.policy_epochs):
            dataloader = DataLoader(self.policy_memory, self.batch_size, shuffle = False)
            for states, actions, rewards, dones, next_states, _ in dataloader:
                self._update_policy_step(states, actions, rewards, dones, next_states)

        states, _, _, _, _ = self.policy_memory.get()
        self.aux_memory.save_all(states)
        self.policy_memory.clear()

    def _update_aux(self) -> None:
        self.policy_old.load_state_dict(self.policy.state_dict())

        for _ in range(self.aux_epochs):
            dataloader = DataLoader(self.aux_memory, self.batch_size, shuffle = False)
            for states in dataloader:
                self._update_aux_step(states)

        self.aux_memory.clear()

    def act(self, state: Tensor) -> Tensor:
        with torch.inference_mode():
            state           = state if self.dont_unsqueeze else state.unsqueeze(0)
            action_datas    = self.policy(state)
            
            if self.is_training_mode:
                action = self.distribution.sample(*action_datas)
            else:
                action = self.distribution.deterministic(*action_datas)

            action = action.squeeze(0)
              
        return action

    def logprob(self, state: Tensor, action: Tensor) -> Tensor:
        with torch.inference_mode():
            state           = state if self.dont_unsqueeze else state.unsqueeze(0)
            action          = action if self.dont_unsqueeze else action.unsqueeze(0)

            action_datas    = self.policy(state)

            logprobs        = self.distribution.logprob(*action_datas, action)
            logprobs        = logprobs.squeeze(0)

        return logprobs

    def save_obs(self, state: Tensor, action: Tensor, reward: Tensor, done: Tensor, next_state: Tensor, logprob: Tensor) -> None:
        self.policy_memory.save(state, action, reward, done, next_state, logprob)

    def save_all(self, states: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor, next_states: Tensor, logprobs: Tensor) -> None:
        self.policy_memory.save_all(states, actions, rewards, dones, next_states, logprobs)
        
    def update(self) -> None:
        self._update_policy()
        self.i_update += 1

        if self.i_update % self.n_aux_update == 0:
            self._update_aux()
            self.i_update = 0 

    def get_obs(self, start_position: int = None, end_position: int = None) -> tuple:
        return self.policy_memory.get(start_position, end_position)

    def clear_obs(self, start_position: int = 0, end_position: int = None) -> None:
        self.policy_memory.clear(start_position, end_position)

    def load_weights(self) -> None:
        model_checkpoint = torch.load(self.folder + '/ppg.tar', map_location = self.device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        
        if self.policy_optimizer is not None:
            self.policy_optimizer.load_state_dict(model_checkpoint['ppo_optimizer_state_dict'])
            
        if self.aux_optimizer is not None:    
            self.aux_optimizer.load_state_dict(model_checkpoint['aux_ppg_optimizer_state_dict'])

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
            'ppo_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'aux_ppg_optimizer_state_dict': self.aux_optimizer.state_dict(),
        }, self.folder + '/ppg.tar')
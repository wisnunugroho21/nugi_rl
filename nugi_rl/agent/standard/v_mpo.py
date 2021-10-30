import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import device

from copy import deepcopy

from nugi_rl.distribution.base import Distribution
from nugi_rl.agent.base import Agent
from nugi_rl.loss.v_mpo.alpha.base import AlphaLoss
from nugi_rl.loss.v_mpo.phi_loss import PhiLoss
from nugi_rl.loss.v_mpo.temperature_loss import TemperatureLoss
from nugi_rl.loss.value import ValueLoss
from nugi_rl.loss.entropy import EntropyLoss
from nugi_rl.memory.policy.base import Memory
from nugi_rl.policy_function.advantage_function.gae import GeneralizedAdvantageEstimation

class AgentVMPO(Agent):
    def __init__(self, policy: Module, value: Module, gae: GeneralizedAdvantageEstimation, distribution: Distribution, alpha_loss: AlphaLoss, phi_loss: PhiLoss, entropy_loss: EntropyLoss, temperature_loss: TemperatureLoss, value_loss: ValueLoss,
            memory: Memory, policy_optimizer: Optimizer, value_optimizer: Optimizer, epochs: int = 10, is_training_mode: bool = True, batch_size: int = 64, folder: str = 'model', 
            device: device = torch.device('cuda:0'), old_policy: Module = None, old_value: Module = None):   

        self.batch_size         = batch_size
        self.epochs             = epochs
        self.is_training_mode   = is_training_mode
        self.folder             = folder

        self.policy             = policy
        self.old_policy         = old_policy
        self.value              = value
        self.old_value          = old_value

        self.distribution       = distribution
        self.memory             = memory
        self.gae                = gae
        
        self.alpha_loss         = alpha_loss
        self.phi_loss           = phi_loss
        self.temperature_loss   = temperature_loss
        self.value_loss         = value_loss
        self.entropy_loss       = entropy_loss        

        self.policy_optimizer   = policy_optimizer
        self.value_optimizer    = value_optimizer   
        self.device             = device

        self.i_update           = 0

        if self.old_policy is None:
            self.old_policy  = deepcopy(self.policy)

        if self.old_value is None:
            self.old_value  = deepcopy(self.value)

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()

    def _update_step(self, states: list, actions: list, rewards: float, dones: bool, next_states: list) -> None:
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        action_datas, temperature, alpha    = self.policy(states)
        old_action_datas, _, _              = self.old_policy(states, True)       
        values                              = self.value(states)
        old_values                          = self.old_value(states, True)
        next_values                         = self.value(next_states, True)

        adv         = self.gae.compute_advantages(rewards, values, next_values, dones).detach()        
        
        phi_loss    = self.phi_loss.compute_loss(action_datas, actions, temperature, adv)
        temp_loss   = self.temperature_loss.compute_loss(temperature, adv)
        alpha_loss  = self.alpha_loss.compute_loss(action_datas, old_action_datas, alpha)
        value_loss  = self.value_loss.compute_loss(values, adv, old_values)
        ent_loss    = self.entropy_loss.compute_loss(action_datas)

        loss    = phi_loss + temp_loss + alpha_loss + value_loss + ent_loss
        loss.backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

    def act(self, state: list) -> list:
        with torch.inference_mode():
            state               = torch.tensor(state).float().to(self.device).unsqueeze(0)
            action_datas, _, _  = self.policy(state)
            
            if self.is_training_mode:
                action = self.distribution.sample(action_datas)
            else:
                action = self.distribution.deterministic(action_datas)

            action = action.squeeze(0).detach().tolist()
              
        return action

    def logprob(self, state: list, action: list) -> list:
        with torch.inference_mode():
            state               = torch.tensor(state).float().to(self.device).unsqueeze(0)
            action              = torch.tensor(action).float().to(self.device).unsqueeze(0)

            action_datas, _, _  = self.policy(state)

            logprobs            = self.distribution.logprob(action_datas, action)
            logprobs            = logprobs.squeeze(0).detach().tolist()

        return logprobs

    def save_obs(self, state: list, action: list, reward: float, done: bool, next_state: list):
        self.memory.save(state, action, reward, done, next_state)
        
    def update(self) -> None:
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_value.load_state_dict(self.value.state_dict())

        for _ in range(self.epochs):
            dataloader = DataLoader(self.memory, self.batch_size, shuffle = False)
            for states, actions, rewards, dones, next_states in dataloader:
                self._update_step(states.to(self.device), actions.to(self.device), rewards.to(self.device), dones.to(self.device), next_states.to(self.device))

        self.memory.clear()

    def get_obs(self, start_position: int = None, end_position: int = None) -> tuple:
        self.memory.get(start_position, end_position)

    def load_weights(self) -> None:
        model_checkpoint = torch.load(self.folder + '/v_mpo.pth', map_location = self.device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        
        if self.policy_optimizer is not None:
            self.policy_optimizer.load_state_dict(model_checkpoint['policy_optimizer_state_dict'])

        if self.value_optimizer is not None:
            self.value_optimizer.load_state_dict(model_checkpoint['value_optimizer_state_dict'])

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
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, self.folder + '/v_mpo.pth')    
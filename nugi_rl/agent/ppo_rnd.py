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
from nugi_rl.loss.rnd_state_predictor import RndStatePredictor
from nugi_rl.memory.policy.base import PolicyMemory
from nugi_rl.memory.rnd import RndMemory
from nugi_rl.policy_function.advantage_function.gae import GeneralizedAdvantageEstimation

from nugi_rl.helpers.math import normalize, count_new_mean, count_new_std

class AgentPPO(Agent):  
    def __init__(self, policy: Module, ex_value: Module, in_value: Module, rnd_predict: Module, rnd_target: Module, distribution: Distribution, 
        gae: GeneralizedAdvantageEstimation, policy_loss: Ppo, value_loss: ValueLoss, entropy_loss: EntropyLoss, rnd_predictor_loss: RndStatePredictor,
        policy_memory: PolicyMemory, rnd_memory: RndMemory, policy_optimizer: Optimizer, rnd_optimizer: Optimizer, ppo_epochs: int = 10, rnd_epochs: int = 10,
        is_training_mode: bool = True, batch_size: int = 32, folder: str = 'model', device: device = torch.device('cuda'), 
        policy_old: Module = None, ex_value_old: Module = None, in_value_old: Module = None) -> None:

        self.batch_size         = batch_size  
        self.ppo_epochs         = ppo_epochs
        self.rnd_epochs         = rnd_epochs
        self.is_training_mode   = is_training_mode
        self.folder             = folder

        self.policy             = policy
        self.policy_old         = policy_old

        self.ex_value           = ex_value
        self.ex_value_old       = ex_value_old

        self.in_value           = in_value
        self.in_value_old       = in_value_old

        self.rnd_predict        = rnd_predict
        self.rnd_target         = rnd_target

        self.distribution       = distribution
        self.gae                = gae

        self.policy_memory      = policy_memory
        self.rnd_memory         = rnd_memory        
        
        self.policy_loss        = policy_loss
        self.value_loss         = value_loss
        self.entropy_loss       = entropy_loss
        self.rnd_predictor_loss = rnd_predictor_loss

        self.policy_optimizer   = policy_optimizer
        self.rnd_optimizer      = rnd_optimizer
        self.device             = device

        if self.policy_old is None:
            self.policy_old     = deepcopy(self.policy)

        if self.ex_value_old is None:
            self.ex_value_old   = deepcopy(self.ex_value)

        if self.in_value_old is None:
            self.in_value_old   = deepcopy(self.in_value)

        if is_training_mode:
          self.policy.train()
          self.ex_value.train()
          self.in_value.train()
        else:
          self.policy.eval()
          self.ex_value.eval()
          self.in_value.eval()

    def _compute_intrinsic_reward(self, obs, mean_obs: Tensor, std_obs: Tensor) -> None:
        obs             = normalize(obs, mean_obs, std_obs)
        
        state_pred      = self.rnd_predict(obs)
        state_target    = self.rnd_target(obs)

        return (state_target - state_pred)

    def _update_obs_normalization_param(self, obs: Tensor) -> None:
        obs                 = torch.FloatTensor(obs).to(self.device)

        mean_obs            = count_new_mean(self.rnd_memory.mean_obs, self.rnd_memory.total_number_obs, obs)
        std_obs             = count_new_std(self.rnd_memory.std_obs, self.rnd_memory.total_number_obs, obs)
        total_number_obs    = len(obs) + self.rnd_memory.total_number_obs
        
        self.rnd_memory.save_observation_normalize_parameter(mean_obs, std_obs, total_number_obs)
    
    def _update_rwd_normalization_param(self, in_rewards: Tensor) -> None:
        std_in_rewards      = count_new_std(self.rnd_memory.std_in_rewards, self.rnd_memory.total_number_rwd, in_rewards)
        total_number_rwd    = len(in_rewards) + self.rnd_memory.total_number_rwd
        
        self.rnd_memory.save_rewards_normalize_parameter(std_in_rewards, total_number_rwd)

    def _update_step_policy(self, states: Tensor, actions: Tensor, ex_rewards: Tensor, dones: Tensor, next_states: Tensor, mean_obs: Tensor, std_obs: Tensor, std_in_rewards: Tensor) -> None:
        self.policy_optimizer.zero_grad()

        action_datas        = self.policy(states)
        old_action_datas    = self.policy_old(states, True)

        ex_values           = self.ex_value(states)
        old_ex_values       = self.ex_value_old(states, True)
        next_ex_values      = self.ex_value(next_states, True)

        in_values           = self.in_value(states)
        old_in_values       = self.in_value_old(states, True)
        next_in_values      = self.in_value(next_states, True)

        obs                 = normalize(next_states, mean_obs, std_obs, 5).detach()
        state_preds         = self.rnd_predict(obs, True)
        state_targets       = self.rnd_target(obs, True)

        in_rewards  = ((state_targets - state_preds).pow(2) * 0.5 / (std_in_rewards.mean() + 1e-6)).detach()

        ex_adv      = self.gae.compute_advantages(ex_rewards, ex_values, next_ex_values, dones).detach()
        in_adv      = self.gae.compute_advantages(in_rewards, in_values, next_in_values, dones).detach()

        loss = 2 * self.policy_loss.compute_loss(action_datas, old_action_datas, actions, ex_adv) + \
            self.policy_loss.compute_loss(action_datas, old_action_datas, actions, in_adv) + \
                self.value_loss.compute_loss(ex_values, ex_adv, old_ex_values) + \
                    self.value_loss.compute_loss(in_values, in_adv, old_in_values)
        
        loss.backward()
        self.policy_optimizer.step()

    def _update_step_rnd(self, obs: Tensor, mean_obs: Tensor, std_obs: Tensor) -> None:
        self.rnd_optimizer.zero_grad()

        obs             = normalize(obs, mean_obs, std_obs)
        
        state_pred      = self.rnd_predict(obs)
        state_target    = self.rnd_target(obs)

        loss            = self.rnd_predictor_loss.compute_loss(state_pred, state_target)
        
        loss.backward()
        self.rnd_optimizer.step()

    def _update_ppo(self) -> None:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.ex_value_old.load_state_dict(self.ex_value.state_dict())
        self.in_value_old.load_state_dict(self.in_value.state_dict())

        for _ in range(self.ppo_epochs):
            dataloader = DataLoader(self.policy_memory, self.batch_size, shuffle = False)
            for states, actions, rewards, dones, next_states, _ in dataloader:
                self._update_step_policy(states, actions, rewards, dones, next_states, self.rnd_memory.mean_obs, self.rnd_memory.std_obs, self.rnd_memory.std_in_rewards)

        self.policy_memory.clear()

    def _update_rnd(self) -> None:
        for _ in range(self.rnd_epochs):
            dataloader  = DataLoader(self.rnd_memory, self.batch_size, shuffle = False)       
            for obs in dataloader:
                self._update_step_rnd(obs.to(self.device), self.rnd_memory.mean_obs.to(self.device), self.rnd_memory.std_obs.to(self.device))

        intrinsic_rewards = self._compute_intrinsic_reward(self.rnd_memory.get().to(self.device), self.rnd_memory.mean_obs.to(self.device), self.rnd_memory.std_obs.to(self.device))
        
        self._update_obs_normalization_param(self.rnd_memory.get())
        self._update_rwd_normalization_param(intrinsic_rewards)

        self.rnd_memory.clear()

    def act(self, state: Tensor) -> Tensor:
        with torch.inference_mode():
            state           = state.unsqueeze(0)
            action_datas    = self.policy(state)
            
            if self.is_training_mode:
                action = self.distribution.sample(action_datas)
            else:
                action = self.distribution.deterministic(action_datas)

            action = action.squeeze(0).detach()
              
        return action

    def logprob(self, state: Tensor, action: Tensor) -> Tensor:
        with torch.inference_mode():
            state           = state.unsqueeze(0)
            action          = action.unsqueeze(0)

            action_datas    = self.policy(state)

            logprobs        = self.distribution.logprob(action_datas, action)
            logprobs        = logprobs.squeeze(0).detach()

        return logprobs

    def save_obs(self, state: Tensor, action: Tensor, reward: Tensor, done: Tensor, next_state: Tensor, logprob: Tensor) -> None:
        self.policy_memory.save(state, action, reward, done, next_state, logprob)
        self.rnd_memory.save(next_state)

    def save_memory(self, memory: PolicyMemory) -> None:
        states, actions, rewards, dones, next_states, logprobs = memory.get()

        self.policy_memory.save_all(states, actions, rewards, dones, next_states, logprobs)
        self.rnd_memory.save_all(next_states)
        
    def update(self, type) -> None:
        if type == 'episodic':
            self._update_ppo()            
        elif type == 'iter':
            self._update_rnd()            
        else:
            raise Exception('choose type update properly (episodic, iter)')

    def get_obs(self, start_position: int = None, end_position: int = None) -> tuple:
        return self.policy_memory.get(start_position, end_position)

    def load_weights(self) -> None:
        model_checkpoint = torch.load(self.folder + '/ppo_rnd.tar', map_location = self.device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.ex_value.load_state_dict(model_checkpoint['ex_value_state_dict'])
        self.in_value.load_state_dict(model_checkpoint['in_value_state_dict'])
        self.rnd_predict.load_state_dict(model_checkpoint['rnd_predict_state_dict'])
        self.rnd_target.load_state_dict(model_checkpoint['rnd_target_state_dict'])
        
        if self.policy_optimizer is not None:
            self.policy_optimizer.load_state_dict(model_checkpoint['policy_optimizer_state_dict'])

        if self.rnd_optimizer is not None:
            self.rnd_optimizer.load_state_dict(model_checkpoint['rnd_optimizer_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.ex_value.train()
            self.in_value.train()
            self.rnd_predict.train()
            self.rnd_target.train()

        else:
            self.policy.eval()
            self.ex_value.eval()
            self.in_value.eval()
            self.rnd_predict.eval()
            self.rnd_target.eval()

    def save_weights(self) -> None:            
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'ex_value_state_dict': self.ex_value.state_dict(),
            'in_value_state_dict': self.in_value.state_dict(),
            'rnd_predict_state_dict': self.rnd_predict.state_dict(),
            'rnd_target_state_dict': self.rnd_target.state_dict(),
            'policy_optimizer_state_dict': self.optimizer.state_dict(),
            'rnd_optimizer_state_dict': self.optimizer.state_dict(),
        }, self.folder + '/ppo_rnd.tar')

from copy import Error, deepcopy
import torch
from torch.utils.data import DataLoader

from helpers.math_function import normalize, count_new_mean, count_new_std

class AgentPpoRnd():  
    def __init__(self, policy, ex_value, in_value, rnd_predict, rnd_target, distribution, ppo_loss, rnd_predictor_loss, ppo_memory, rnd_memory, ppo_optimizer, rnd_optimizer, 
            ppo_epochs = 10, rnd_epochs = 10, is_training_mode = True, clip_norm = 5, batch_size = 32,  folder = 'model', device = torch.device('cuda:0'), policy_old = None, ex_value_old = None, in_value_old = None):   

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
        
        self.ppo_memory         = ppo_memory
        self.rnd_memory         = rnd_memory        
        
        self.ppo_loss           = ppo_loss
        self.rnd_predictor_loss = rnd_predictor_loss

        self.ppo_optimizer      = ppo_optimizer
        self.rnd_optimizer      = rnd_optimizer

        self.distribution       = distribution
        self.device             = device
        self.clip_norm          = clip_norm

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

    @property
    def memory(self):
        return self.ppo_memory

    def _compute_intrinsic_reward(self, obs, mean_obs, std_obs):
        obs             = normalize(obs, mean_obs, std_obs)
        
        state_pred      = self.rnd_predict(obs)
        state_target    = self.rnd_target(obs)

        return (state_target - state_pred)

    def _update_obs_normalization_param(self, obs):
        obs                 = torch.FloatTensor(obs).to(self.device)

        mean_obs            = count_new_mean(self.rnd_memory.mean_obs, self.rnd_memory.total_number_obs, obs)
        std_obs             = count_new_std(self.rnd_memory.std_obs, self.rnd_memory.total_number_obs, obs)
        total_number_obs    = len(obs) + self.rnd_memory.total_number_obs
        
        self.rnd_memory.save_observation_normalize_parameter(mean_obs, std_obs, total_number_obs)
    
    def _update_rwd_normalization_param(self, in_rewards):
        std_in_rewards      = count_new_std(self.rnd_memory.std_in_rewards, self.rnd_memory.total_number_rwd, in_rewards)
        total_number_rwd    = len(in_rewards) + self.rnd_memory.total_number_rwd
        
        self.rnd_memory.save_rewards_normalize_parameter(std_in_rewards, total_number_rwd)

    def _training_ppo(self, states, actions, ex_rewards, dones, next_states, mean_obs, std_obs, std_in_rewards): 
        action_datas        = self.policy(states)
        old_action_datas    = self.policy_old(states, True)

        ex_values           = self.ex_value(states)
        old_ex_values       = self.ex_value_old(states, True)
        next_ex_values      = self.ex_value(next_states, True)

        in_values           = self.in_value(states)
        old_in_values       = self.in_value_old(states, True)
        next_in_values      = self.in_value(next_states, True)

        obs                 = normalize(next_states, mean_obs, std_obs, self.clip_norm).detach()
        state_preds         = self.rnd_predict(obs)
        state_targets       = self.rnd_target(obs)

        loss = self.ppo_loss.compute_loss(action_datas, old_action_datas, ex_values, old_ex_values, next_ex_values, actions, ex_rewards, dones,
        state_preds, state_targets, in_values, old_in_values, next_in_values, std_in_rewards)

        self.ppo_optimizer.zero_grad()
        loss.backward()
        self.ppo_optimizer.step()

    def _training_rnd(self, obs, mean_obs, std_obs):
        obs             = normalize(obs, mean_obs, std_obs)
        
        state_pred      = self.rnd_predict(obs)
        state_target    = self.rnd_target(obs)

        loss            = self.rnd_predictor_loss.compute_loss(state_pred, state_target)

        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()

    def _update_ppo(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.ex_value_old.load_state_dict(self.ex_value.state_dict())
        self.in_value_old.load_state_dict(self.in_value.state_dict())

        for _ in range(self.ppo_epochs):
            dataloader = DataLoader(self.ppo_memory, self.batch_size, shuffle = False)
            for states, actions, rewards, dones, next_states in dataloader:
                self._training_ppo(states.to(self.device), actions.to(self.device), rewards.to(self.device), dones.to(self.device), next_states.to(self.device),
                    self.rnd_memory.mean_obs.to(self.device), self.rnd_memory.std_obs.to(self.device), self.rnd_memory.std_in_rewards.to(self.device))

        self.ppo_memory.clear_memory()

    def _update_rnd(self):
        dataloader  = DataLoader(self.rnd_memory, self.batch_size, shuffle = False)

        for _ in range(self.rnd_epochs):       
            for obs in dataloader:
                self._training_rnd(obs.to(self.device), self.rnd_memory.mean_obs.to(self.device), self.rnd_memory.std_obs.to(self.device))

        intrinsic_rewards = self._compute_intrinsic_reward(self.rnd_memory.get_all_tensor().to(self.device), self.rnd_memory.mean_obs.to(self.device), self.rnd_memory.std_obs.to(self.device))
        
        self._update_obs_normalization_param(self.rnd_memory.observations)
        self._update_rwd_normalization_param(intrinsic_rewards)

        self.rnd_memory.clear_memory()           

    def update(self, type):
        if type == 'episodic':
            self._update_ppo()            
        elif type == 'iter':
            self._update_rnd()            
        else:
            raise Error('choose type update properly')

    def act(self, state):
        state           = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_datas    = self.policy(state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.deterministic(action_datas)
              
        return action.squeeze().detach().tolist()

    def logprobs(self, state, action):
        state           = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action          = torch.FloatTensor(action).unsqueeze(0).to(self.device)

        action_datas    = self.policy(state)
        logprobs        = self.distribution.logprob(action_datas, action)

        return logprobs.squeeze().detach().tolist()

    def save_obs(self, state, action, reward, done, next_state):
        self.ppo_memory.save_obs(state, action, reward, done, next_state)
        self.rnd_memory.save_obs(next_state)

    def save_weights(self, folder = None):
        if folder == None:
            folder = self.folder
            
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
        }, self.folder + '/ppg.tar')
        
    def load_weights(self, folder = None, device = None):
        if device == None:
            device = self.device

        if folder == None:
            folder = self.folder

        model_checkpoint = torch.load(self.folder + '/ppg.tar', map_location = device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        self.ppo_optimizer.load_state_dict(model_checkpoint['ppo_optimizer_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.value.train()

        else:
            self.policy.eval()
            self.value.eval()

    def get_weights(self):
        return self.policy.state_dict(), self.value.state_dict()

    def set_weights(self, policy_weights, value_weights):
        self.policy.load_state_dict(policy_weights)
        self.value.load_state_dict(value_weights)
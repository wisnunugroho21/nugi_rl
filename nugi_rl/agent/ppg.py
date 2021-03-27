import copy

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from helpers.pytorch_utils import set_device, to_numpy, to_tensor

class AgentPPG():  
    def __init__(self, policy, value, state_dim, action_dim, distribution, ppo_loss, aux_ppg_loss, ppo_memory, aux_ppg_memory, 
                ppo_optimizer, aux_ppg_optimizer, PPO_epochs = 10, Aux_epochs = 10, n_aux_update = 10, is_training_mode = True, policy_kl_range = 0.03, 
                policy_params = 5, value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, batch_size = 32,  folder = 'model', use_gpu = True):   

        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.batch_size         = batch_size  
        self.PPO_epochs         = PPO_epochs
        self.Aux_epochs         = Aux_epochs
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.state_dim          = state_dim
        self.folder             = folder
        self.use_gpu            = use_gpu
        self.n_aux_update       = n_aux_update

        self.policy             = policy
        self.policy_old         = copy.deepcopy(self.policy)

        self.value              = value
        self.value_old          = copy.deepcopy(self.value)

        self.distribution       = distribution
        self.ppo_memory         = ppo_memory
        self.aux_memory         = aux_ppg_memory
        
        self.ppoLoss            = ppo_loss
        self.auxLoss            = aux_ppg_loss      

        self.device             = set_device(self.use_gpu)
        self.i_update           = 0

        self.ppo_optimizer      = ppo_optimizer
        self.aux_ppg_optimizer  = aux_ppg_optimizer

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()

    def __training_ppo(self, states, actions, rewards, dones, next_states):         
        self.ppo_optimizer.zero_grad()

        action_datas, _     = self.policy(states)
        values              = self.value(states)

        old_action_datas, _ = self.policy_old(states, True)
        old_values          = self.value_old(states, True)
        next_values         = self.value(next_states, True)

        loss = self.ppoLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)
        loss.backward()

        self.ppo_optimizer.step()

    def __training_aux_ppg(self, states):        
        self.aux_ppg_optimizer.zero_grad()
        
        action_datas, values    = self.policy(states)

        returns                 = self.value(states, True)
        old_action_datas, _     = self.policy_old(states, True)

        loss = self.auxLoss.compute_loss(action_datas, old_action_datas, values, returns)
        loss.backward()

        self.aux_ppg_optimizer.step()

    def _update_policy(self):
        dataloader = DataLoader(self.ppo_memory, self.batch_size, shuffle = False, num_workers = 2)

        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in dataloader:
                self.__training_ppo(to_tensor(states, use_gpu = self.use_gpu), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu))

        states, _, _, _, _ = self.ppo_memory.get_all_items()
        self.aux_memory.save_all(states)
        self.ppo_memory.clear_memory()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())    

    def _update_aux_ppg(self):
        dataloader  = DataLoader(self.aux_memory, self.batch_size, shuffle = False, num_workers = 2)

        for _ in range(self.Aux_epochs):       
            for states in dataloader:
                self.__training_aux_ppg(to_tensor(states, use_gpu = self.use_gpu))

        self.aux_memory.clear_memory()
        self.policy_old.load_state_dict(self.policy.state_dict())    

    def update(self):
        self._update_policy()
        self.i_update += 1

        if self.i_update % self.n_aux_update == 0:
            self._update_aux_ppg()
            self.i_update = 0

    def save_memory(self, ppo_memory):
        states, actions, rewards, dones, next_states = ppo_memory.get_all_items()
        self.ppo_memory.save_all(states, actions, rewards, dones, next_states)

    def act(self, state):
        state           = to_tensor(state, use_gpu = self.use_gpu, first_unsqueeze = True, detach = True)
        action_datas, _ = self.policy(state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.deterministic(action_datas)
              
        return to_numpy(action, self.use_gpu)

    def save_weights(self, folder = None):
        if folder == None:
            folder = self.folder
            
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
            'auxppg_optimizer_state_dict': self.aux_ppg_optimizer.state_dict()
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
        self.aux_ppg_optimizer.load_state_dict(model_checkpoint['auxppg_optimizer_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.value.train()
            print('Model is training...')

        else:
            self.policy.eval()
            self.value.eval()
            print('Model is evaluating...')

    def get_weights(self):
        return self.policy.state_dict(), self.value.state_dict()

    def set_weights(self, policy_weights, value_weights):
        self.policy.load_state_dict(policy_weights)
        self.value.load_state_dict(value_weights)
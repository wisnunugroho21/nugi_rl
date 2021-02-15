import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import numpy as np

from memory.list_memory import ListMemory
from memory.aux_memory import AuxMemory

from utils.pytorch_utils import set_device, to_numpy

class Agent():  
    def __init__(self, Policy_Model, Value_Model, state_dim, action_dim, distribution, policy_loss, aux_loss, policy_memory, aux_memory,
                is_training_mode = True, policy_kl_range = 0.03, policy_params = 5, 
                value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, 
                batch_size = 32, PPO_epochs = 10, Aux_epochs = 10, gamma = 0.99,
                lam = 0.95, learning_rate = 3e-4, folder = 'model', use_gpu = True):   

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
        self.learning_rate      = learning_rate
        self.folder             = folder
        self.use_gpu            = use_gpu

        self.device             = set_device(self.use_gpu)

        self.policy             = Policy_Model(state_dim, action_dim, self.use_gpu)
        self.policy_old         = Policy_Model(state_dim, action_dim, self.use_gpu)

        self.value              = Value_Model(state_dim, action_dim, self.use_gpu)
        self.value_old          = Value_Model(state_dim, action_dim, self.use_gpu)
        
        self.ppo_optimizer      = Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr = learning_rate)        
        self.aux_optimizer      = Adam(self.policy.parameters(), lr = learning_rate)

        self.distribution       = distribution
        self.policy_memory      = policy_memory
        self.aux_memory         = aux_memory
        
        self.trulyPPO           = policy_loss
        self.auxLoss            = aux_loss

        self.scaler             = torch.cuda.amp.GradScaler()

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()

    def save_eps(self, state, action, reward, done, next_state):
        self.policy_memory.save_eps(state, action, reward, done, next_state)

    def save_all(self, states, actions, rewards, dones, next_states):
        self.policy_memory.save_all(states, actions, rewards, dones, next_states)

    def save_memory(self, memory):
        states, actions, rewards, dones, next_states = memory.get_all_items()
        self.policy_memory.save_all(states, actions, rewards, dones, next_states)

    def act(self, state):
        if isinstance(state, tuple):
            state = list(state)
            for i, s in enumerate(list(state)):
                s           = torch.FloatTensor(s).to(self.device)
                state[i]    = s.unsqueeze(0).detach() if len(s.shape) == 1 or len(s.shape) == 3 else s.detach()
            state = tuple(state)            
        else:
            state   = torch.FloatTensor(state).to(self.device)
            state   = state.unsqueeze(0).detach() if len(state.shape) == 1 or len(state.shape) == 3 else state.detach()
        
        action_datas, _ = self.policy(state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.act_deterministic(action_datas)
              
        return to_numpy(action, self.use_gpu)

    def training_ppo(self, states, actions, rewards, dones, next_states): 
        self.ppo_optimizer.zero_grad()

        action_datas, _     = self.policy(states)
        values              = self.value(states)
        old_action_datas, _ = self.policy_old(states, True)
        old_values          = self.value_old(states, True)
        next_values         = self.value(next_states, True)

        with torch.cuda.amp.autocast():
            ppo_loss    = self.trulyPPO.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)

        self.scaler.scale(ppo_loss).backward()
        self.scaler.step(self.ppo_optimizer)
        self.scaler.update()

    def training_aux(self, states):
        self.aux_optimizer.zero_grad()
        
        action_datas, values    = self.policy(states)
        returns                 = self.value(states, True)
        old_action_datas, _     = self.policy_old(states, True)

        with torch.cuda.amp.autocast():
            joint_loss  = self.auxLoss.compute_loss(action_datas, old_action_datas, values, returns)

        self.scaler.scale(joint_loss).backward()
        self.scaler.step(self.aux_optimizer)
        self.scaler.update()

    def update_ppo(self, policy_memory = None, aux_memory = None):
        if policy_memory is None:
            policy_memory = self.policy_memory

        if aux_memory is None:
            aux_memory = self.aux_memory

        dataloader = DataLoader(policy_memory, self.batch_size, shuffle = False)

        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in dataloader:
                if isinstance(states, list) and isinstance(next_states, list):
                    for i, (s, ns) in enumerate(zip(states, next_states)):
                        states[i], next_states[i]   = torch.FloatTensor(s).to(self.device), torch.FloatTensor(ns).to(self.device)
                    states, next_states = tuple(states), tuple(next_states)
                else:
                    states      = states.float().to(self.device)
                    next_states = next_states.float().to(self.device)

                self.training_ppo(states, actions.float().to(self.device), rewards.float().to(self.device), dones.float().to(self.device), next_states)

        states, _, _, _, _ = policy_memory.get_all_items()
        aux_memory.save_all(states)
        policy_memory.clear_memory()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

        return policy_memory, aux_memory

    def update_aux(self, aux_memory = None):
        if aux_memory is None:
            aux_memory = self.aux_memory

        dataloader  = DataLoader(aux_memory, self.batch_size, shuffle = False)

        for _ in range(self.Aux_epochs):       
            for states in dataloader:
                if isinstance(states, tuple):
                    for i, s in enumerate(states):
                        states[i]   = torch.FloatTensor(s).to(self.device)
                else:
                    states  = states.float().to(self.device)

                self.training_aux(states.float().to(self.device))

        aux_memory.clear_memory()
        self.policy_old.load_state_dict(self.policy.state_dict())

        return aux_memory

    def save_weights(self):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.ppo_optimizer.state_dict()
            }, self.folder + '/policy.tar')
        
        torch.save({
            'model_state_dict': self.value.state_dict(),
            'optimizer_state_dict': self.aux_optimizer.state_dict()
            }, self.folder + '/value.tar')
        
    def load_weights(self):
        policy_checkpoint = torch.load(self.folder + '/policy.tar')
        self.policy.load_state_dict(policy_checkpoint['model_state_dict'])
        self.ppo_optimizer.load_state_dict(policy_checkpoint['optimizer_state_dict'])

        value_checkpoint = torch.load(self.folder + '/value.tar')
        self.value.load_state_dict(value_checkpoint['model_state_dict'])
        self.aux_optimizer.load_state_dict(value_checkpoint['optimizer_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.value.train()
            print('Model is training...')

        else:
            self.policy.eval()
            self.value.eval()
            print('Model is evaluating...')

    def save_temp_weights(self):
        torch.save(self.policy.state_dict(), 'agent_policy.pth')
        torch.save(self.value.state_dict(), 'agent_value.pth')

    def load_temp_weights(self, device = None):
        if device == None:
            device = self.device

        self.policy.load_state_dict(torch.load('agent_policy.pth', map_location = device))
        self.value.load_state_dict(torch.load('agent_value.pth', map_location = device))

    def get_weights(self):
        return self.policy.state_dict(), self.value.state_dict()

    def set_weights(self, policy_weights, value_weights):
        self.policy.load_state_dict(policy_weights)
        self.value.load_state_dict(value_weights)
import torch
import torchvision.transforms as transforms

from torch.utils import data
from torch.utils.data import DataLoader, dataloader
from torch.optim import Adam

import numpy as np
from distribution.basic import BasicContinous
from loss.other.kl import KL

from utils.pytorch_utils import set_device, to_numpy, to_tensor
class AgentPpgVae():  
    def __init__(self, Policy_Model, Value_Model, CnnModel, DecoderModel, state_dim, action_dim, policy_dist, vae_dist, policy_loss, aux_loss, vae_loss, 
                policy_memory, aux_memory, vae_memory, PPO_epochs = 10, Aux_epochs = 10, Vae_epochs = 10, n_ppo_update = 1024, n_aux_update = 10, 
                is_training_mode = True, policy_kl_range = 0.03, policy_params = 5, value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, 
                batch_size = 32,  learning_rate = 3e-4, folder = 'model', use_gpu = True):   

        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.batch_size         = batch_size  
        self.PPO_epochs         = PPO_epochs
        self.Aux_epochs         = Aux_epochs
        self.Vae_epochs         = Vae_epochs
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.state_dim          = state_dim
        self.learning_rate      = learning_rate
        self.folder             = folder
        self.use_gpu            = use_gpu
        self.n_aux_update       = n_aux_update
        self.n_ppo_update       = n_ppo_update

        self.policy             = Policy_Model(state_dim, action_dim, self.use_gpu)
        self.policy_old         = Policy_Model(state_dim, action_dim, self.use_gpu)
        self.policy_cnn         = CnnModel()
        self.policy_cnn_old     = CnnModel()
        self.policy_decoder     = DecoderModel()

        self.value              = Value_Model(state_dim, self.use_gpu)
        self.value_old          = Value_Model(state_dim, self.use_gpu)
        self.value_cnn          = CnnModel()
        self.value_cnn_old      = CnnModel()
        self.value_decoder      = DecoderModel()

        self.policy_dist        = policy_dist
        self.vae_dist           = vae_dist

        self.policy_memory      = policy_memory
        self.aux_memory         = aux_memory
        self.vae_memory         = vae_memory
        
        self.policyLoss         = policy_loss
        self.auxLoss            = aux_loss
        self.vaeLoss            = vae_loss
        self.klLoss             = KL(BasicContinous(use_gpu))

        self.device             = set_device(self.use_gpu)
        self.i_aux_update       = 0
        self.i_ppo_update       = 0

        self.ppo_optimizer      = Adam(list(self.policy.parameters()) + list(self.policy_cnn.parameters()) + list(self.value.parameters()) + list(self.value_cnn.parameters()), lr = learning_rate)               
        self.aux_optimizer      = Adam(list(self.policy.parameters()) + list(self.policy_cnn.parameters()), lr = learning_rate)
        self.vae_pol_optimizer  = Adam(list(self.policy_cnn.parameters()) + list(self.policy_decoder.parameters()), lr = learning_rate) 
        self.vae_val_optimizer  = Adam(list(self.value_cnn.parameters()) + list(self.value_decoder.parameters()), lr = learning_rate)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())
        self.policy_cnn_old.load_state_dict(self.policy_cnn.state_dict())
        self.value_cnn_old.load_state_dict(self.value_cnn.state_dict())       

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()

    def __training_ppo(self, states, actions, rewards, dones, next_states):         
        self.ppo_optimizer.zero_grad()

        out_mean1, out_std1 = self.policy_cnn(states)
        action_datas, _     = self.policy(self.vae_dist.sample((out_mean1, out_std1)).mean([-1, -2]))

        out_mean2, out_std2 = self.value_cnn(states)
        values              = self.value(self.vae_dist.sample((out_mean2, out_std2)).mean([-1, -2]))

        out_mean3, out_std3 = self.policy_cnn_old(states, True)
        old_action_datas, _ = self.policy_old(self.vae_dist.sample((out_mean3, out_std3)).mean([-1, -2]), True)

        out_mean4, out_std4 = self.value_cnn_old(states, True)
        old_values          = self.value_old(self.vae_dist.sample((out_mean4, out_std4)).mean([-1, -2]), True)

        out_mean5, out_std5 = self.value_cnn(next_states, True)
        next_values         = self.value(self.vae_dist.sample((out_mean5, out_std5)).mean([-1, -2]), True)

        zeros, ones = torch.zeros_like(out_mean1), torch.ones_like(out_std1)

        loss = self.policyLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones) + self.klLoss.compute_loss(out_mean1, out_std1, zeros, ones) + self.klLoss.compute_loss(out_mean2, out_std2, zeros, ones)
        loss.backward()

        self.ppo_optimizer.step()

    def __training_aux(self, states):        
        self.aux_optimizer.zero_grad()
        
        out_mean1, out_std1     = self.policy_cnn(states)
        action_datas, values    = self.policy(self.vae_dist.sample((out_mean1, out_std1)).mean([-1, -2]))

        out_mean2, out_std2     = self.value_cnn(states, True)
        returns                 = self.value(self.vae_dist.sample((out_mean2, out_std2)).mean([-1, -2]), True)

        out_mean3, out_std3     = self.policy_cnn_old(states, True)
        old_action_datas, _     = self.policy_old(self.vae_dist.sample((out_mean3, out_std3)).mean([-1, -2]), True)

        zeros, ones = torch.zeros_like(out_mean1), torch.ones_like(out_std1)

        loss = self.auxLoss.compute_loss(action_datas, old_action_datas, values, returns) + self.klLoss.compute_loss(out_mean1, out_std1, zeros, ones)
        loss.backward()

        self.aux_optimizer.step()

    def __training_vae(self, states):
        self.vae_pol_optimizer.zero_grad()

        out_mean1, out_std1     = self.policy_cnn(states)
        reconstruc_states1      = self.policy_decoder(self.vae_dist.sample((out_mean1, out_std1)))
        
        zeros, ones             = torch.zeros_like(out_mean1), torch.ones_like(out_std1)        

        loss = self.vaeLoss.compute_loss(states, reconstruc_states1, out_mean1, out_std1, zeros, ones)
        loss.backward()
        self.vae_pol_optimizer.step()

        self.vae_val_optimizer.zero_grad()

        out_mean2, out_std2     = self.value_cnn(states)
        reconstruc_states2      = self.value_decoder(self.vae_dist.sample((out_mean2, out_std2)))

        zeros, ones = torch.zeros_like(out_mean1), torch.ones_like(out_std1)

        loss = self.vaeLoss.compute_loss(states, reconstruc_states2, out_mean2, out_std2, zeros, ones)
        loss.backward()
        self.vae_val_optimizer.step()

    def __update_ppo(self):
        dataloader = DataLoader(self.policy_memory, self.batch_size, shuffle = False, num_workers = 2)

        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in dataloader:
                self.__training_ppo(to_tensor(states, use_gpu = self.use_gpu), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu))

        states, _, _, _, _ = self.policy_memory.get_all_items()
        self.aux_memory.save_all(states)
        self.policy_memory.clear_memory()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())
        self.policy_cnn_old.load_state_dict(self.policy_cnn.state_dict())
        self.value_cnn_old.load_state_dict(self.value_cnn.state_dict())

    def __update_aux(self):
        dataloader  = DataLoader(self.aux_memory, self.batch_size, shuffle = False, num_workers = 2)

        for _ in range(self.Aux_epochs):       
            for states in dataloader:
                self.__training_aux(to_tensor(states, use_gpu = self.use_gpu))

        self.aux_memory.clear_memory()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_cnn_old.load_state_dict(self.policy_cnn.state_dict())

    def __update_vae(self):
        if len(self.vae_memory) >= self.batch_size:
            for _ in range(self.Vae_epochs):
                dataloader  = DataLoader(self.vae_memory, self.batch_size, shuffle = True, num_workers = 2)
                inputs      = next(iter(dataloader))

                self.__training_vae(to_tensor(inputs, use_gpu = self.use_gpu))

        self.policy_cnn_old.load_state_dict(self.policy_cnn.state_dict())
        self.value_cnn_old.load_state_dict(self.value_cnn.state_dict())

    def act(self, state):
        state               = to_tensor(state, use_gpu = self.use_gpu, first_unsqueeze = True, detach = True)

        out_mean1, out_std1 = self.policy_cnn(state)
        action_datas, _     = self.policy(self.vae_dist.sample((out_mean1, out_std1)).mean([-1, -2]))
        
        if self.is_training_mode:
            action = self.policy_dist.sample(action_datas)
        else:
            action = self.policy_dist.deterministic(action_datas)
              
        return to_numpy(action, self.use_gpu)

    def save_memory(self, policy_memory):
        states, actions, rewards, dones, next_states = policy_memory.get_all_items()
        self.policy_memory.save_all(states, actions, rewards, dones, next_states)
        self.vae_memory.save_all(states)    

    def update(self):
        self.__update_vae()
        self.i_ppo_update += 1      

        if self.i_ppo_update % self.n_ppo_update == 0 and self.i_ppo_update != 0:
            self.__update_ppo()
            self.i_ppo_update = 0
            self.i_aux_update += 1

        if self.i_aux_update % self.n_aux_update == 0 and self.i_aux_update != 0:
            self.__update_aux()
            self.i_aux_update = 0

    def save_weights(self):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.ppo_optimizer.state_dict()
            }, self.folder + '/policy.tar')
        
        torch.save({
            'model_state_dict': self.value.state_dict(),
            'optimizer_state_dict': self.aux_optimizer.state_dict()
            }, self.folder + '/value.tar')
        
    def load_weights(self, device = None):
        if device == None:
            device = self.device

        policy_checkpoint = torch.load(self.folder + '/policy.tar', map_location = device)
        self.policy.load_state_dict(policy_checkpoint['model_state_dict'])
        self.ppo_optimizer.load_state_dict(policy_checkpoint['optimizer_state_dict'])

        value_checkpoint = torch.load(self.folder + '/value.tar', map_location = device)
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

    def get_weights(self):
        return self.policy.state_dict(), self.value.state_dict()

    def set_weights(self, policy_weights, value_weights):
        self.policy.load_state_dict(policy_weights)
        self.value.load_state_dict(value_weights)
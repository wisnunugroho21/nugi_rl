import copy

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from helpers.pytorch_utils import set_device, to_numpy, to_tensor
from agent.ppg import AgentPPG

class AgentImageStatePPG(AgentPPG):
    def __init__(self, cnn, policy, value, state_dim, action_dim, distribution, ppo_loss, aux_ppg_loss, ppo_memory, aux_ppg_memory, 
            ppo_optimizer, aux_ppg_optimizer, PPO_epochs = 10, Aux_epochs = 10, n_aux_update = 10, is_training_mode = True, policy_kl_range = 0.03, 
            policy_params = 5, value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, batch_size = 32,  folder = 'model', use_gpu = True):

        super().__init__(policy, value, state_dim, action_dim, distribution, ppo_loss, aux_ppg_loss, ppo_memory, aux_ppg_memory, 
            ppo_optimizer, aux_ppg_optimizer, PPO_epochs, Aux_epochs, n_aux_update, is_training_mode, policy_kl_range, 
            policy_params, value_clip, entropy_coef, vf_loss_coef, batch_size,  folder, use_gpu)

        self.cnn = cnn

    def _training_ppo(self, data_states, actions, rewards, dones, data_next_states):  
        images, states              = data_states
        next_images, next_states    = data_next_states

        self.ppo_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            res                 = self.cnn(images)
            action_datas, _     = self.policy(res, states)
            values              = self.value(res, states)

            old_action_datas, _ = self.policy_old(res, states, True)
            old_values          = self.value_old(res, states, True)

            next_res            = self.cnn(next_images, True)
            next_values         = self.value(next_res, next_states, True)

            loss = self.ppoLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)
        
        self.ppo_scaler.scale(loss).backward()
        self.ppo_scaler.step(self.ppo_optimizer)
        self.ppo_scaler.update()

    def _training_aux_ppg(self, data_states):
        images, states  = data_states

        self.aux_ppg_optimizer.zero_grad()        
        with torch.cuda.amp.autocast():
            res                     = self.cnn(images, True)

            action_datas, values    = self.policy(res, states)

            returns                 = self.value(res, states, True)
            old_action_datas, _     = self.policy_old(res, states, True)

            loss = self.auxLoss.compute_loss(action_datas, old_action_datas, values, returns)

        self.auxppg_scaler.scale(loss).backward()
        self.auxppg_scaler.step(self.aux_ppg_optimizer)
        self.auxppg_scaler.update()

    def act(self, data_state):
        image, state        = data_state
        image, state        = torch.FloatTensor(self.trans(image)).unsqueeze(0).to(self.device), torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        res                 = self.cnn(image)
        action_datas, _     = self.policy(res, state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.deterministic(action_datas)
              
        return to_numpy(action, self.use_gpu)
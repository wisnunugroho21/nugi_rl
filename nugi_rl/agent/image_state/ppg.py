import copy

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from helpers.pytorch_utils import set_device, to_list, to_tensor
from agent.standard.ppg import AgentPPG

class AgentImageStatePPG(AgentPPG):
    def __init__(self, cnn, policy, value, state_dim, action_dim, distribution, ppo_loss, aux_ppg_loss, ppo_memory, aux_ppg_memory, 
            ppo_optimizer, aux_ppg_optimizer, PPO_epochs = 10, Aux_epochs = 10, n_aux_update = 10, is_training_mode = True, policy_kl_range = 0.03, 
            policy_params = 5, value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, batch_size = 32,  folder = 'model', use_gpu = True):

        super().__init__(policy, value, state_dim, action_dim, distribution, ppo_loss, aux_ppg_loss, ppo_memory, aux_ppg_memory, 
            ppo_optimizer, aux_ppg_optimizer, PPO_epochs, Aux_epochs, n_aux_update, is_training_mode, policy_kl_range, 
            policy_params, value_clip, entropy_coef, vf_loss_coef, batch_size,  folder, use_gpu)

        self.cnn        = cnn
        self.cnn_old    = copy.deepcopy(self.cnn)

        if self.is_training_mode:
            self.cnn.train()
        else:
            self.cnn.eval()

    def _training_ppo(self, images, states, actions, rewards, dones, next_images, next_states):
        res                 = self.cnn(images)

        action_datas, _     = self.policy(res, states)
        values              = self.value(res, states)
        
        res_old             = self.cnn_old(images, True)

        old_action_datas, _ = self.policy_old(res_old, states, True)
        old_values          = self.value_old(res_old, states, True)

        next_res            = self.cnn(next_images, True)
        next_values         = self.value(next_res, next_states, True)

        loss = self.ppoLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)
        
        self.ppo_optimizer.zero_grad()
        loss.backward()
        self.ppo_optimizer.step()

    def _training_aux_ppg(self, images, states):
        res                     = self.cnn(images, True)

        returns                 = self.value(res, states, True)
        old_action_datas, _     = self.policy_old(res, states, True)
        
        action_datas, values    = self.policy(res, states)            

        loss = self.auxLoss.compute_loss(action_datas, old_action_datas, values, returns)

        self.aux_ppg_optimizer.zero_grad()
        loss.backward()
        self.aux_ppg_optimizer.step()

    def _update_ppo(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())
        self.cnn_old.load_state_dict(self.cnn.state_dict())        

        for _ in range(self.ppo_epochs):
            dataloader = DataLoader(self.ppo_memory, self.batch_size, shuffle = False, num_workers = 8)       
            for images, states, actions, rewards, dones, next_images, next_states in dataloader: 
                self._training_ppo(images.float().to(self.device), states.float().to(self.device), actions.float().to(self.device), rewards.float().to(self.device), dones.float().to(self.device), next_images.float().to(self.device), next_states.float().to(self.device))

        images, states, _, _, _, _, _ = self.ppo_memory.get_all_items()
        self.aux_ppg_memory.save_all(images, states)
        self.ppo_memory.clear_memory()
        
    def _update_aux_ppg(self):
        self.policy_old.load_state_dict(self.policy.state_dict())        

        for _ in range(self.aux_ppg_epochs):
            dataloader  = DataLoader(self.aux_ppg_memory, self.batch_size, shuffle = False, num_workers = 8)       
            for images, states in dataloader:
                self._training_aux_ppg(images.float().to(self.device), states.float().to(self.device))

        self.aux_ppg_memory.clear_memory()        

    def act(self, image, state):
        image, state        = self.ppo_memory.transform(image).unsqueeze(0).to(self.device), torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        res                 = self.cnn(image)
        action_datas, _     = self.policy(res, state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.deterministic(action_datas)
              
        return action.squeeze().detach().tolist()

    def logprobs(self, image, state, action):
        image, state    = self.ppo_memory.transform(image).unsqueeze(0).to(self.device), torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action          = torch.FloatTensor(action).unsqueeze(0).float().to(self.device)

        res             = self.cnn(image)
        action_datas, _ = self.policy(res, state)

        logprobs        = self.distribution.logprob(action_datas, action)
        return logprobs.squeeze().detach().tolist()

    def save_obs(self, image, state, action, reward, done, next_image, next_state):
        self.ppo_memory.save_obs(image, state, action, reward, done, next_image, next_state)

    def save_weights(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'cnn_state_dict': self.cnn.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
            'aux_ppg_optimizer_state_dict': self.aux_ppg_optimizer.state_dict(),
        }, self.folder + '/ppg.tar')
        
    def load_weights(self, device = None):
        if device == None:
            device = self.device

        model_checkpoint = torch.load(self.folder + '/ppg.tar', map_location = device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        self.cnn.load_state_dict(model_checkpoint['cnn_state_dict'])
        self.ppo_optimizer.load_state_dict(model_checkpoint['ppo_optimizer_state_dict'])        
        self.aux_ppg_optimizer.load_state_dict(model_checkpoint['aux_ppg_optimizer_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.value.train()
            self.cnn.train()

        else:
            self.policy.eval()
            self.value.eval()
            self.cnn.eval()
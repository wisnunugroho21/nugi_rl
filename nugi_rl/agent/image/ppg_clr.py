import copy

import torch
from torch.utils.data import DataLoader

from helpers.pytorch_utils import set_device, to_list

class AgentPPGClr():  
    def __init__(self, projector, cnn, policy, value, state_dim, action_dim, distribution, ppo_loss, aux_ppg_loss, aux_clr_loss, ppo_memory, aux_ppg_memory, aux_clr_memory,
            ppo_optimizer, aux_ppg_optimizer, aux_clr_optimizer, ppo_epochs = 10, aux_ppg_epochs = 10, aux_clr_epochs = 10, n_aux_update = 10, is_training_mode = True, policy_kl_range = 0.03, 
            policy_params = 5, value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, batch_size = 32,  folder = 'model', use_gpu = True):   

        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.batch_size         = batch_size  
        self.ppo_epochs         = ppo_epochs
        self.aux_ppg_epochs     = aux_ppg_epochs
        self.aux_clr_epochs     = aux_clr_epochs
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

        self.cnn                = cnn
        self.cnn_old            = copy.deepcopy(self.cnn)

        self.projector          = projector
        self.projector_old      = copy.deepcopy(self.projector)

        self.distribution       = distribution
        self.device             = set_device(self.use_gpu)
        self.i_update           = 0

        self.ppo_memory         = ppo_memory
        self.aux_ppg_memory     = aux_ppg_memory
        self.aux_clr_memory     = aux_clr_memory
        
        self.ppoLoss            = ppo_loss
        self.aux_ppg_loss       = aux_ppg_loss
        self.aux_clrLoss        = aux_clr_loss

        self.ppo_optimizer      = ppo_optimizer
        self.aux_ppg_optimizer  = aux_ppg_optimizer
        self.aux_clr_optimizer  = aux_clr_optimizer

        self.ppo_scaler         = torch.cuda.amp.GradScaler()
        self.aux_ppg_scaler     = torch.cuda.amp.GradScaler()
        self.aux_clr_scaler     = torch.cuda.amp.GradScaler()

        if is_training_mode:
            self.policy.train()
            self.value.train()
            self.cnn.train()
            self.projector.train()
        else:
            self.policy.eval()
            self.value.eval()
            self.cnn.eval()
            self.projector.eval()

    @property
    def memory(self):
        return self.ppo_memory

    def _training_ppo(self, states, actions, rewards, dones, next_states):
        self.ppo_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            res                 = self.cnn(states)

            action_datas, _     = self.policy(res)
            values              = self.value(res)
            
            res_old             = self.cnn_old(states, True)

            old_action_datas, _ = self.policy_old(res_old, True)
            old_values          = self.value_old(res_old, True)

            next_res            = self.cnn(next_states, True)
            next_values         = self.value(next_res, True)

            loss = self.ppoLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)
        
        self.ppo_scaler.scale(loss).backward()
        self.ppo_scaler.step(self.ppo_optimizer)
        self.ppo_scaler.update()

    def _training_aux_ppg(self, states):
        self.aux_ppg_optimizer.zero_grad()        
        with torch.cuda.amp.autocast():
            res                     = self.cnn(states, True)

            returns                 = self.value(res, True)
            old_action_datas, _     = self.policy_old(res, True)

            action_datas, values    = self.policy(res)                        

            loss = self.auxLoss.compute_loss(action_datas, old_action_datas, values, returns)

        self.aux_ppg_scaler.scale(loss).backward()
        self.aux_ppg_scaler.step(self.aux_ppg_optimizer)
        self.aux_ppg_scaler.update()

    def _training_aux_clr(self, input_images, target_images):
        self.aux_clr_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            res_anchor        = self.cnn(input_images)
            encoded_anchor    = self.projector(res_anchor)

            res_target        = self.cnn_old(target_images, True)
            encoded_target    = self.projector_old(res_target, True)

            loss = self.aux_clrLoss.compute_loss(encoded_anchor, encoded_target)

        self.aux_clr_scaler.scale(loss).backward()
        self.aux_clr_scaler.step(self.aux_clr_optimizer)
        self.aux_clr_scaler.update()

    def _update_ppo(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())
        self.cnn_old.load_state_dict(self.cnn.state_dict()) 

        for _ in range(self.ppo_epochs): 
            dataloader = DataLoader(self.ppo_memory, self.batch_size, shuffle = False, num_workers = 8)
            for states, actions, rewards, dones, next_states in dataloader:
                self._training_ppo(states.float().to(self.device), actions.float().to(self.device), rewards.float().to(self.device), dones.float().to(self.device), next_states.float().to(self.device))

        states, _, _, _, _ = self.ppo_memory.get_all_items()
        self.aux_ppg_memory.save_all(states)
        self.ppo_memory.clear_memory()           

    def _update_aux_ppg(self):
        self.policy_old.load_state_dict(self.policy.state_dict())

        for _ in range(self.aux_ppg_epochs):
            dataloader  = DataLoader(self.aux_ppg_memory, self.batch_size, shuffle = False, num_workers = 8)       
            for states in dataloader:
                self._training_aux_ppg(states.float().to(self.device))

        self.aux_ppg_memory.clear_memory()

    def _update_aux_clr(self):
        self.cnn_old.load_state_dict(self.cnn.state_dict())
        self.projector_old.load_state_dict(self.projector.state_dict())

        for _ in range(self.aux_clr_epochs):
            dataloader  = DataLoader(self.aux_clr_memory, self.batch_size, shuffle = True, num_workers = 8)
            for input_images, target_images in dataloader:
                self._training_aux_clr(input_images.to(self.device), target_images.to(self.device))            

        self.aux_clr_memory.clear_memory()

    def update(self):
        self._update_ppo()
        self.i_update += 1

        if self.i_update % self.n_aux_update == 0:
            self._update_aux_ppg()
            self._update_aux_clr()
            self.i_update = 0

    def act(self, state):
        state           = self.ppo_memory.transform(state).unsqueeze(0).to(self.device)

        res             = self.cnn(state)
        action_datas, _ = self.policy(res)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.deterministic(action_datas)
              
        return to_list(action.squeeze(), self.use_gpu)

    def logprobs(self, state, action):
        state           = self.ppo_memory.transform(state).unsqueeze(0).to(self.device)
        action          = torch.FloatTensor(action).unsqueeze(0).float().to(self.device)

        res             = self.cnn(state)
        action_datas, _ = self.policy(res)

        logprobs        = self.distribution.logprob(action_datas, action)
        return logprobs.squeeze().detach().tolist()

    def save_obs(self, state, action, reward, done, next_state):
        self.ppo_memory.save_obs(state, action, reward, done, next_state)

    def save_weights(self, folder = None):
        if folder == None:
            folder = self.folder
            
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'cnn_state_dict': self.cnn.state_dict(),
            'projector_state_dict': self.projector.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
            'aux_ppg_optimizer_state_dict': self.aux_ppg_optimizer.state_dict(),
            'aux_clr_optimizer_state_dict': self.aux_clr_optimizer.state_dict(),
            'ppo_scaler_state_dict': self.ppo_scaler.state_dict(),
            'aux_ppg_scaler_state_dict': self.aux_ppg_scaler.state_dict(),
            'aux_clr_scaler_state_dict': self.aux_clr_scaler.state_dict(),
        }, self.folder + '/ppg_clr.tar')
        
    def load_weights(self, folder = None, device = None):
        if device == None:
            device = self.device

        if folder == None:
            folder = self.folder

        model_checkpoint = torch.load(self.folder + '/ppg_clr.tar', map_location = device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        self.cnn.load_state_dict(model_checkpoint['cnn_state_dict'])
        self.projector.load_state_dict(model_checkpoint['projector_state_dict'])
        self.ppo_optimizer.load_state_dict(model_checkpoint['ppo_optimizer_state_dict'])        
        self.aux_ppg_optimizer.load_state_dict(model_checkpoint['aux_ppg_optimizer_state_dict'])   
        self.aux_clr_optimizer.load_state_dict(model_checkpoint['aux_clr_optimizer_state_dict'])
        self.ppo_scaler.load_state_dict(model_checkpoint['ppo_scaler_state_dict'])        
        self.aux_ppg_scaler.load_state_dict(model_checkpoint['aux_ppg_scaler_state_dict'])  
        self.aux_clr_scaler.load_state_dict(model_checkpoint['aux_clr_scaler_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.value.train()
            self.cnn.train()
            self.projector.train()

        else:
            self.policy.eval()
            self.value.eval()
            self.cnn.eval()
            self.projector.eval()

    def get_weights(self):
        return self.policy.state_dict(), self.value.state_dict()

    def set_weights(self, policy_weights, value_weights):
        self.policy.load_state_dict(policy_weights)
        self.value.load_state_dict(value_weights)
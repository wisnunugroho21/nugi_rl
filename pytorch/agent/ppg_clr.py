import torch
import torchvision.transforms as transforms

from torch.utils import data
from torch.utils.data import DataLoader, dataloader
from torch.optim import Adam

import numpy as np

from utils.pytorch_utils import set_device, to_numpy, to_tensor
class AgentPpgClr():  
    def __init__(self, Policy_Model, Value_Model, CnnModel, ProjectionModel, state_dim, action_dim, policy_dist, policy_loss, aux_loss, clr_loss, 
                policy_memory, aux_memory, clr_memory, PPO_epochs = 10, Aux_epochs = 10, Clr_epochs = 10, n_ppo_update = 1024, n_aux_update = 10, 
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
        self.Clr_epochs         = Clr_epochs
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
        self.policy_projection  = ProjectionModel()

        self.value              = Value_Model(state_dim, self.use_gpu)
        self.value_old          = Value_Model(state_dim, self.use_gpu)
        self.value_cnn          = CnnModel()
        self.value_cnn_old      = CnnModel()
        self.value_projection   = ProjectionModel()

        self.policy_dist        = policy_dist

        self.policy_memory      = policy_memory
        self.aux_memory         = aux_memory
        self.clr_memory         = clr_memory
        
        self.policyLoss         = policy_loss
        self.auxLoss            = aux_loss
        self.clrLoss            = clr_loss

        self.device             = set_device(self.use_gpu)
        self.i_aux_update       = 0
        self.i_ppo_update       = 0

        self.ppo_optimizer      = Adam(list(self.policy_cnn.parameters()) + list(self.policy.parameters()) + list(self.value_cnn.parameters()) + list(self.value.parameters()), lr = learning_rate)        
        self.aux_optimizer      = Adam(list(self.policy_cnn.parameters()) + list(self.policy.parameters()), lr = learning_rate)
        self.clr_optimizer      = Adam(list(self.policy_cnn.parameters()) + list(self.policy_projection.parameters()) + list(self.value_cnn.parameters()) + list(self.value_projection.parameters()), lr = learning_rate) 

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

        out1                = self.policy_cnn(states)
        action_datas, _     = self.policy(out1.mean([-1, -2]))

        out2                = self.value_cnn(states)
        values              = self.value(out2.mean([-1, -2]))

        out3                = self.policy_cnn_old(states, True)
        old_action_datas, _ = self.policy_old(out3.mean([-1, -2]), True)

        out4                = self.value_cnn_old(states, True)
        old_values          = self.value_old(out4.mean([-1, -2]), True)

        out5                = self.value_cnn(next_states, True)
        next_values         = self.value(out5.mean([-1, -2]), True)

        loss = self.policyLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)
        loss.backward()

        self.ppo_optimizer.step()

    def __training_aux(self, states):        
        self.aux_optimizer.zero_grad()
        
        out1                    = self.policy_cnn(states)
        action_datas, values    = self.policy(out1.mean([-1, -2]))

        out2                    = self.value_cnn(states, True)
        returns                 = self.value(out2.mean([-1, -2]), True)

        out3                    = self.policy_cnn_old(states, True)
        old_action_datas, _     = self.policy_old(out3.mean([-1, -2]), True)

        loss = self.auxLoss.compute_loss(action_datas, old_action_datas, values, returns)
        loss.backward()

        self.aux_optimizer.step()

    def __training_clr(self, first_inputs, second_inputs):
        self.clr_optimizer.zero_grad()

        out1            = self.policy_cnn(first_inputs)
        first_encoded   = self.policy_projection(out1.mean([-1, -2]))

        out2            = self.value_cnn(second_inputs)
        second_encoded  = self.value_projection(out2.mean([-1, -2]))

        loss = self.clrLoss.compute_loss(first_encoded, second_encoded)
        loss.backward()
        self.clr_optimizer.step()

        self.clr_optimizer.zero_grad()

        out1            = self.value_cnn(first_inputs)
        first_encoded   = self.value_projection(out1.mean([-1, -2]))

        out2            = self.policy_cnn(second_inputs)
        second_encoded  = self.policy_projection(out2.mean([-1, -2]))

        loss = self.clrLoss.compute_loss(first_encoded, second_encoded)
        loss.backward()
        self.clr_optimizer.step()

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

    def __update_clr(self):
        dataloader  = DataLoader(self.clr_memory, self.batch_size, shuffle = True, num_workers = 2)

        for _ in range(self.Clr_epochs):
            for first_inputs, second_inputs in dataloader:
                self.__training_clr(to_tensor(first_inputs, use_gpu = self.use_gpu), to_tensor(second_inputs, use_gpu = self.use_gpu))

        self.clr_memory.clear_memory()
        
        self.policy_cnn_old.load_state_dict(self.policy_cnn.state_dict())
        self.value_cnn_old.load_state_dict(self.value_cnn.state_dict())

    def act(self, state):
        state               = to_tensor(state, use_gpu = self.use_gpu, first_unsqueeze = True, detach = True)

        out1                = self.policy_cnn(state)
        action_datas, _     = self.policy(out1.mean([-1, -2]))
        
        if self.is_training_mode:
            action = self.policy_dist.sample(action_datas)
        else:
            action = self.policy_dist.deterministic(action_datas)
              
        return to_numpy(action, self.use_gpu)

    def save_memory(self, policy_memory):
        states, actions, rewards, dones, next_states = policy_memory.get_all_items()
        self.policy_memory.save_all(states, actions, rewards, dones, next_states)
        self.clr_memory.save_all(states)

    def update(self):
        self.__update_ppo()
        self.__update_clr()        
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
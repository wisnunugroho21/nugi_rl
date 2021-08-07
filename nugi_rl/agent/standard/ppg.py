
from copy import deepcopy
import torch
from torch.utils.data import DataLoader

class AgentPPG():  
    def __init__(self, policy, value, distribution, ppo_loss, aux_ppg_loss, ppo_memory, aux_ppg_memory, 
                ppo_optimizer, aux_ppg_optimizer, ppo_epochs = 10, aux_ppg_epochs = 10, n_aux_update = 10, is_training_mode = True, 
                batch_size = 32,  folder = 'model', device = torch.device('cuda:0')):   

        self.batch_size         = batch_size  
        self.ppo_epochs         = ppo_epochs
        self.aux_ppg_epochs     = aux_ppg_epochs
        self.is_training_mode   = is_training_mode
        self.folder             = folder
        self.n_aux_update       = n_aux_update

        self.policy             = policy
        self.policy_old         = deepcopy(self.policy)

        self.value              = value
        self.value_old          = deepcopy(self.value)

        self.distribution       = distribution
        self.ppo_memory         = ppo_memory
        self.aux_ppg_memory     = aux_ppg_memory
        
        self.ppoLoss            = ppo_loss
        self.auxLoss            = aux_ppg_loss      

        self.device             = device
        self.i_update           = 0

        self.ppo_optimizer      = ppo_optimizer
        self.aux_ppg_optimizer  = aux_ppg_optimizer

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()

    @property
    def memory(self):
        return self.ppo_memory

    def _training_ppo(self, states, actions, rewards, dones, next_states): 
        action_datas, _     = self.policy(states)
        values              = self.value(states)

        old_action_datas, _ = self.policy_old(states, True)
        old_values          = self.value_old(states, True)
        next_values         = self.value(next_states, True)

        loss = self.ppoLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)

        self.ppo_optimizer.zero_grad()
        loss.backward()
        self.ppo_optimizer.step()

    def _training_aux_ppg(self, states):  
        action_datas, values    = self.policy(states)

        returns                 = self.value(states, True)
        old_action_datas, _     = self.policy_old(states, True)

        loss = self.auxLoss.compute_loss(action_datas, old_action_datas, values, returns)

        self.aux_ppg_optimizer.zero_grad()
        loss.backward()
        self.aux_ppg_optimizer.step()

    def _update_ppo(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

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

    def update(self):
        self._update_ppo()
        self.i_update += 1

        if self.i_update % self.n_aux_update == 0:
            self._update_aux_ppg()
            self.i_update = 0

    def save_memory(self, ppo_memory):
        states, actions, rewards, dones, next_states = ppo_memory.get_all_items()
        self.ppo_memory.save_all(states, actions, rewards, dones, next_states)

    def act(self, state):
        state           = torch.FloatTensor(state).unsqueeze(0).float().to(self.device)
        action_datas, _ = self.policy(state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.deterministic(action_datas)
              
        return action.squeeze().detach().tolist()

    def save_weights(self, folder = None):
        if folder == None:
            folder = self.folder
            
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
            'aux_ppg_optimizer_state_dict': self.aux_ppg_optimizer.state_dict(),
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
        self.aux_ppg_optimizer.load_state_dict(model_checkpoint['aux_ppg_optimizer_state_dict'])

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
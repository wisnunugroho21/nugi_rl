import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.pytorch_utils import set_device, to_numpy, to_tensor

class AgentPpoOff():
    def __init__(self, Policy_Model, Value_Model, Q_Model, state_dim, action_dim, distribution, on_policy_loss, off_policy_loss, off_value_loss,
                on_memory, off_memory, is_training_mode = True, policy_kl_range = 0.03, policy_params = 5, value_clip = 1.0, entropy_coef = 0.0, 
                vf_loss_coef = 1.0, batch_size = 32, On_epochs = 10, Off_epochs = 1, learning_rate = 3e-4, folder = 'model', use_gpu = True):

        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.batch_size         = batch_size  
        self.On_epochs          = On_epochs
        self.Off_epochs         = Off_epochs
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.state_dim          = state_dim
        self.learning_rate      = learning_rate
        self.folder             = folder
        self.use_gpu            = use_gpu

        self.policy             = Policy_Model(state_dim, action_dim, self.use_gpu)
        self.policy_old         = Policy_Model(state_dim, action_dim, self.use_gpu)

        self.value              = Value_Model(state_dim, self.use_gpu)
        self.value_old          = Value_Model(state_dim, self.use_gpu)

        self.soft_q             = Q_Model(state_dim, action_dim, self.use_gpu)
        self.soft_q_old         = Q_Model(state_dim, action_dim, self.use_gpu)

        self.distribution       = distribution

        self.onMemory           = on_memory
        self.offMemory          = off_memory
        
        self.onPolicyLoss       = on_policy_loss
        self.offPolicyLoss      = off_policy_loss
        self.offValueLoss       = off_value_loss

        self.device             = set_device(self.use_gpu)
        self.i_update           = 0
        
        self.ppo_on_optimizer   = Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr = learning_rate)
        self.ppo_off_optimizer  = Adam(list(self.policy.parameters()) + list(self.soft_q.parameters()), lr = learning_rate)
        self.value_optimizer    = Adam(self.policy.parameters(), lr = learning_rate) 

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict()) 
        self.soft_q_old.load_state_dict(self.soft_q.state_dict()) 

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()

    def __training_off_policy(self, states, actions, rewards, dones, next_states):
        self.ppo_off_optimizer.zero_grad()

        action_datas, _     = self.policy(states)
        old_action_datas, _ = self.policy_old(states, True)

        values              = self.soft_q(states, actions)
        old_values          = self.soft_q_old(states, actions, True)
        next_values         = self.value(next_states, True)

        loss = self.offPolicyLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)
        loss.backward()
        
        self.ppo_off_optimizer.step()

    def __training_off_values(self, states, rewards, dones, next_states):
        self.value_optimizer.zero_grad()

        values              = self.value(states)
        old_values          = self.value_old(states, True)
        next_values         = self.value(next_states, True)

        loss = self.offValueLoss.compute_loss(values, old_values, next_values, rewards, dones)
        loss.backward()        

        self.value_optimizer.step()

    def __training_on_policy(self, states, actions, rewards, dones, next_states):         
        self.ppo_on_optimizer.zero_grad()

        action_datas, _     = self.policy(states)
        values              = self.value(states)
        old_action_datas, _ = self.policy_old(states, True)
        old_values          = self.value_old(states, True)
        next_values         = self.value(next_states, True)

        loss = self.onPolicyLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)
        loss.backward()
        
        self.ppo_on_optimizer.step()

    def update_off(self):
        dataloader = DataLoader(self.offMemory, self.batch_size, shuffle = False)

        for _ in range(self.Off_epochs):       
            for states, actions, rewards, dones, next_states in dataloader:
                self.__training_off_policy(to_tensor(states, use_gpu = self.use_gpu), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu))
                    
                self.__training_off_values(to_tensor(states, use_gpu = self.use_gpu), rewards.float().to(self.device), 
                    dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu))

        self.offMemory.clear_memory()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict()) 
        self.soft_q_old.load_state_dict(self.soft_q.state_dict()) 

    def update_on(self):
        dataloader = DataLoader(self.onMemory, self.batch_size, shuffle = False)

        for _ in range(self.On_epochs):       
            for states, actions, rewards, dones, next_states in dataloader:
                self.__training_on_policy(to_tensor(states, use_gpu = self.use_gpu), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu))                

        self.onMemory.clear_memory()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

    def act(self, state):
        state               = to_tensor(state, use_gpu = self.use_gpu, first_unsqueeze = True, detach = True)
        action_datas        = self.policy(state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.act_deterministic(action_datas)
              
        return to_numpy(action, self.use_gpu)

    def save_on_eps(self, state, action, reward, done, next_state):
        self.onPolicyMemory.save_eps(state, action, reward, done, next_state)

    def save_off_eps(self, state, action, reward, done, next_state):
        self.offMemory.save_eps(state, action, reward, done, next_state)

    def save_on_memory(self, policy_memory):
        states, actions, rewards, dones, next_states = policy_memory.get_all_items()
        self.onPolicyMemory.save_all(states, actions, rewards, dones, next_states)

    def save_off_memory(self, policy_memory):
        states, actions, rewards, dones, next_states = policy_memory.get_all_items()
        self.offMemory.save_all(states, actions, rewards, dones, next_states)

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
            print('Model is evaluating...')s
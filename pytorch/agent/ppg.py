import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.pytorch_utils import set_device, to_numpy, to_tensor

class AgentPPG():  
    def __init__(self, Policy_Model, Value_Model, state_dim, action_dim, distribution, policy_loss, aux_loss, policy_memory, aux_memory,
                is_training_mode = True, policy_kl_range = 0.03, policy_params = 5, value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, 
                batch_size = 32, PPO_epochs = 10, Aux_epochs = 10, learning_rate = 3e-4, folder = 'model', use_gpu = True, n_aux_update = 10):   

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
        self.n_aux_update       = n_aux_update

        self.policy             = Policy_Model(state_dim, action_dim, self.use_gpu)
        self.policy_old         = Policy_Model(state_dim, action_dim, self.use_gpu)

        self.value              = Value_Model(state_dim, self.use_gpu)
        self.value_old          = Value_Model(state_dim, self.use_gpu)

        self.distribution       = distribution
        self.policy_memory      = policy_memory
        self.aux_memory         = aux_memory
        
        self.policyLoss         = policy_loss
        self.auxLoss            = aux_loss      

        self.device             = set_device(self.use_gpu)
        self.i_update           = 0

        self.ppo_optimizer      = Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr = learning_rate)        
        self.aux_optimizer      = Adam(self.policy.parameters(), lr = learning_rate)

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

        loss = self.policyLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)
        loss.backward()

        self.ppo_optimizer.step()

    def __training_aux(self, states):        
        self.aux_optimizer.zero_grad()
        
        action_datas, values    = self.policy(states)
        returns                 = self.value(states, True)
        old_action_datas, _     = self.policy_old(states, True)

        loss = self.auxLoss.compute_loss(action_datas, old_action_datas, values, returns)
        loss.backward()

        self.aux_optimizer.step()

    def __update_ppo(self):
        dataloader = DataLoader(self.policy_memory, self.batch_size, shuffle = False)

        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in dataloader:
                self.__training_ppo(to_tensor(states, use_gpu = self.use_gpu), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu))

        states, _, _, _, _ = self.policy_memory.get_all_items()
        self.aux_memory.save_all(states)
        self.policy_memory.clear_memory()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())    

    def __update_aux(self):
        dataloader  = DataLoader(self.aux_memory, self.batch_size, shuffle = False)

        for _ in range(self.Aux_epochs):       
            for states in dataloader:
                self.__training_aux(to_tensor(states, use_gpu = self.use_gpu))

        self.aux_memory.clear_memory()
        self.policy_old.load_state_dict(self.policy.state_dict())    

    def update(self):
        self.__update_ppo()
        self.i_update += 1

        if self.i_update % self.n_aux_update == 0:
            self.__update_aux()
            self.i_update = 0

    def save_eps(self, state, action, reward, done, next_state):
        self.policy_memory.save_eps(state, action, reward, done, next_state)

    def save_memory(self, policy_memory):
        states, actions, rewards, dones, next_states = policy_memory.get_all_items()
        self.policy_memory.save_all(states, actions, rewards, dones, next_states)

    def act(self, state):
        state           = to_tensor(state, use_gpu = self.use_gpu, first_unsqueeze = True, detach = True)
        action_datas, _ = self.policy(state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.act_deterministic(action_datas)
              
        return to_numpy(action, self.use_gpu)

    def save_weights(self, folder = None):
        if folder == None:
            folder = self.folder
            
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.ppo_optimizer.state_dict()
            }, folder + '/policy.tar')
        
        torch.save({
            'model_state_dict': self.value.state_dict(),
            'optimizer_state_dict': self.aux_optimizer.state_dict()
            }, folder + '/value.tar')
        
    def load_weights(self, folder = None, device = None):
        if device == None:
            device = self.device

        if folder == None:
            folder = self.folder

        policy_checkpoint = torch.load(folder + '/policy.tar', map_location = device)
        self.policy.load_state_dict(policy_checkpoint['model_state_dict'])
        self.ppo_optimizer.load_state_dict(policy_checkpoint['optimizer_state_dict'])

        value_checkpoint = torch.load(folder + '/value.tar', map_location = device)
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
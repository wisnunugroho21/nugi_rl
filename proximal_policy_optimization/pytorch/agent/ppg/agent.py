import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from memory.list_memory import ListMemory
from memory.aux_memory import AuxMemory

from utils.pytorch_utils import set_device

class Agent():  
    def __init__(self, Policy_Model, Value_Model, state_dim, action_dim,
                is_training_mode = True, policy_kl_range = 0.0008, policy_params = 20, 
                value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, 
                batch_size = 32, PPO_epochs = 4, Aux_epochs = 4, gamma = 0.99, 
                lam = 0.95, learning_rate = 2.5e-4, folder = 'model', use_gpu = True):   

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

        self.policy             = Policy_Model(state_dim, action_dim)
        self.policy_old         = Policy_Model(state_dim, action_dim)
        self.policy_optimizer   = Adam(self.policy.parameters(), lr = learning_rate)

        self.value              = Value_Model(state_dim, action_dim)
        self.value_old          = Value_Model(state_dim, action_dim)
        self.value_optimizer    = Adam(self.value.parameters(), lr = learning_rate)

        self.policy_memory      = ListMemory()
        self.aux_memory         = AuxMemory()

        self.device             = set_device(use_gpu)
        self.use_gpu            = use_gpu

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()

    def set_params(self, params):
        self.value_clip         = self.value_clip * params if self.value_clip is not None else self.value_clip
        self.policy_kl_range    = self.policy_kl_range * params
        self.policy_params      = self.policy_params * params

    def save_eps(self, state, action, reward, done, next_state):
        self.policy_memory.save_eps(state, action, reward, done, next_state)

    def save_all(self, states, actions, rewards, dones, next_states):
        self.policy_memory.save_all(states, actions, rewards, dones, next_states)

    def act(self, state):
        pass

    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, rewards, dones, next_states): 
        pass

    def training_aux(self, states):
        pass

    # Update the model
    def update_ppo(self):
        dataloader = DataLoader(self.policy_memory, self.batch_size, shuffle = False)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in dataloader: 
                self.training_ppo(states.float().to(self.device), actions.float().to(self.device), \
                    rewards.float().to(self.device), dones.float().to(self.device), next_states.float().to(self.device))

        # Clear the memory
        states, _, _, _, _ = self.policy_memory.get_all_items()
        self.aux_memory.save_all(states)
        self.policy_memory.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

    def update_aux(self):
        dataloader  = DataLoader(self.aux_memory, self.batch_size, shuffle = False)

        # Optimize policy for K epochs:
        for _ in range(self.Aux_epochs):       
            for states in dataloader:
                self.training_aux(states.float().to(self.device))

        # Clear the memory
        self.aux_memory.clear_memory()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save_weights(self):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.policy_optimizer.state_dict()
            }, self.folder + '/policy.tar')
        
        torch.save({
            'model_state_dict': self.value.state_dict(),
            'optimizer_state_dict': self.value_optimizer.state_dict()
            }, self.folder + '/value.tar')
        
    def load_weights(self):
        policy_checkpoint = torch.load(self.folder + '/policy.tar')
        self.policy.load_state_dict(policy_checkpoint['model_state_dict'])
        self.policy_optimizer.load_state_dict(policy_checkpoint['optimizer_state_dict'])

        value_checkpoint = torch.load(self.folder + '/value.tar')
        self.value.load_state_dict(value_checkpoint['model_state_dict'])
        self.value_optimizer.load_state_dict(value_checkpoint['optimizer_state_dict'])

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
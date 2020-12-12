import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from memory.list_memory import ListMemory
from utils.pytorch_utils import set_device

class Agent:  
    def __init__(self, Actor_Model, Critic_Model, state_dim, action_dim,
                is_training_mode = True, policy_kl_range = 0.0008, policy_params = 20, 
                value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, 
                batch_size = 32, PPO_epochs = 4, gamma = 0.99, 
                lam = 0.95, learning_rate = 2.5e-4, folder = 'model', use_gpu = True):   

        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.batch_size         = batch_size  
        self.PPO_epochs         = PPO_epochs
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.learning_rate      = learning_rate
        self.folder             = folder

        self.actor              = Actor_Model(state_dim, action_dim, use_gpu)
        self.actor_old          = Actor_Model(state_dim, action_dim, use_gpu)
        self.actor_optimizer    = Adam(self.actor.parameters(), lr = learning_rate)

        self.critic             = Critic_Model(state_dim, action_dim, use_gpu)
        self.critic_old         = Critic_Model(state_dim, action_dim, use_gpu)
        self.critic_optimizer   = Adam(self.critic.parameters(), lr = learning_rate)

        self.memory             = ListMemory()
        self.device             = set_device(use_gpu)
        self.use_gpu            = use_gpu

        if is_training_mode:
            self.actor.train()
            self.critic.train()
            print('Model is training...')

        else:
            self.actor.eval()
            self.critic.eval()
            print('Model is evaluating...')

    def set_params(self, params):
        self.value_clip         = self.value_clip * params if self.value_clip is not None else self.value_clip
        self.policy_kl_range    = self.policy_kl_range * params
        self.policy_params      = self.policy_params * params

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.save_eps(state, action, reward, done, next_state)

    def save_all(self, states, actions, rewards, dones, next_states):
        self.memory.save_all(states, actions, rewards, dones, next_states)

    def act(self, state):
        pass

    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, rewards, dones, next_states): 
        pass

    # Update the model
    def update_ppo(self):
        dataloader = DataLoader(self.memory, self.batch_size, shuffle = False)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in dataloader: 
                self.training_ppo(states.float().to(self.device), actions.float().to(self.device), \
                    rewards.float().to(self.device), dones.float().to(self.device), next_states.float().to(self.device))

        # Clear the memory
        self.memory.clear_memory()

        # Copy new weights into old policy:
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def update_model_ppo(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def save_weights(self):
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict()
            }, self.folder + '/actor.tar')
        
        torch.save({
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict()
            }, self.folder + '/critic.tar')
        
    def load_weights(self):
        actor_checkpoint = torch.load(self.folder + '/actor.tar')
        self.actor.load_state_dict(actor_checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])

        critic_checkpoint = torch.load(self.folder + '/critic.tar')
        self.critic.load_state_dict(critic_checkpoint['model_state_dict'])
        self.critic_optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])

        if self.is_training_mode:
            self.actor.train()
            self.critic.train()
            print('Model is training...')

        else:
            self.actor.eval()
            self.critic.eval()
            print('Model is evaluating...')

    def get_weights(self):
        return self.actor.state_dict(), self.critic.state_dict()

    def set_weights(self, actor_weights, critic_weights):
        self.actor.load_state_dict(actor_weights)
        self.critic.load_state_dict(critic_weights)
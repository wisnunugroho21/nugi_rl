import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from model.EmbeddingActionNN import Actor_Model, Critic_Model
from memory.on_policy_embedding_act_memory import Memory
from ppo_loss.truly_ppo_discrete import get_loss
from distribution.categorical_distribution import sample
from utils.pytorch_utils import set_device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:  
    def __init__(self, state_dim, action_dim, is_training_mode = True, policy_kl_range = 0.0008, policy_params = 20, value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0,
                 minibatch = 4, PPO_epochs = 4, gamma = 0.99, lam = 0.95, learning_rate = 2.5e-4, folder = 'model', use_gpu = True):        
        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.minibatch          = minibatch       
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

        self.memory             = Memory()
        self.device             = set_device(use_gpu)
        self.use_gpu            = use_gpu

    def set_params(self, params):
        self.value_clip         = self.value_clip * params
        self.policy_kl_range    = self.policy_kl_range * params
        self.policy_params      = self.policy_params * params

    def save_eps(self, state, reward, action, done, next_state, available_action):
        self.memory.save_eps(state, reward, action, done, next_state, available_action) 

    def act(self, state, available_action):
        if len(state.shape) > 1:
            state           = torch.FloatTensor(state).flatten().unsqueeze(0).to(self.device).detach()                
        else:
            state           = torch.FloatTensor(state).unsqueeze(0).to(self.device).detach()
        
        available_action    = torch.LongTensor(available_action).unsqueeze(0).to(self.device).detach()
        action_probs        = self.actor(state, available_action)
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action = sample(action_probs, self.use_gpu)
        else:
            action = torch.argmax(action_probs, 1)  
              
        return action

    # Get loss and Do backpropagation
    def training_ppo(self, states, rewards, actions, dones, next_states, available_actions):      
        action_probs, values          = self.actor(states, available_actions), self.critic(states)
        old_action_probs, old_values  = self.actor_old(states, available_actions), self.critic_old(states)
        next_values                   = self.critic(next_states)

        loss = get_loss(action_probs, old_action_probs, values, old_values, next_values, actions, rewards, dones,
                self.policy_kl_range, self.policy_params, self.value_clip, self.vf_loss_coef, self.entropy_coef, self.use_gpu)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss.backward()

        self.actor_optimizer.step() 
        self.critic_optimizer.step() 

    # Update the model
    def update_ppo(self):        
        batch_size = int(len(self.memory) / self.minibatch)
        dataloader = DataLoader(self.memory, batch_size, shuffle = False)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for states, rewards, actions, dones, next_states, available_actions in dataloader: 
                self.training_ppo(states.float().to(self.device), actions.float().to(self.device), rewards.float().to(self.device), dones.float().to(self.device), next_states.float().to(self.device), available_actions.float().to(self.device))

        # Clear the memory
        self.memory.clear_memory()

        # Copy new weights into old policy:
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
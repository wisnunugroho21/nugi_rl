import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from model.BasicTanhNN import Actor_Model, Critic_Model
from memory.on_policy_impala_memory import OnMemory
from ppo_loss.truly_ppo_continous_impala import get_loss
from distribution.normal_distribution import sample, logprob
from utils.pytorch_utils import set_device, to_numpy

class Agent:  
    def __init__(self, state_dim, action_dim, is_training_mode = True, policy_kl_range = 0.03, policy_params = 5, value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0,
                 minibatch = 32, PPO_epochs = 10, gamma = 0.99, lam = 0.95, learning_rate = 3e-4, action_std = 1.0, folder = 'model', use_gpu = True): 
        
        self.action_std = torch.ones([1, action_dim]).float().to(set_device(use_gpu))

        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.minibatch          = minibatch       
        self.PPO_epochs         = PPO_epochs
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim 
        self.action_std         = self.action_std * torch.FloatTensor([[action_std]]).to(set_device(use_gpu))
        self.learning_rate      = learning_rate   
        self.folder             = folder  

        self.actor              = Actor_Model(state_dim, action_dim, use_gpu)
        self.actor_old          = Actor_Model(state_dim, action_dim, use_gpu)
        self.actor_optimizer    = Adam(self.actor.parameters(), lr = learning_rate)

        self.critic             = Critic_Model(state_dim, action_dim, use_gpu)
        self.critic_old         = Critic_Model(state_dim, action_dim, use_gpu)
        self.critic_optimizer   = Adam(self.critic.parameters(), lr = learning_rate)

        self.memory             = OnMemory()
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
        self.action_std         = self.action_std * params

    def save_eps(self, state, action, reward, done, next_state, logprob, next_next_state = None):
        self.memory.save_eps(state, action, reward, done, next_state, logprob, next_next_state) 

    def save_replace_all_eps(self, states, actions, rewards, dones, next_states, logprobs, next_next_states = None):
        self.memory.save_replace_all(states, actions, rewards, dones, next_states, logprobs, next_next_states)        

    def get_eps(self):
        return self.memory.get_all_items()

    def clearMemory(self):
        return self.memory.clearMemory()

    def convert_next_states_to_next_next_states(self):
        self.memory.convert_next_states_to_next_next_states()

    def act(self, state, need_logprobs = False):
        state           = torch.FloatTensor(state).unsqueeze(0).to(self.device).detach()  
        action_mean     = self.actor(state)
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            
            # Sample the action
            action          = sample(action_mean, self.action_std, self.use_gpu)

            if need_logprobs:
                act_logprob     = logprob(action_mean, self.action_std, action, self.use_gpu)
                return to_numpy(action.squeeze(0).detach()), to_numpy(act_logprob.squeeze(0).detach())
        else:
            action = action_mean  
              
        return to_numpy(action.squeeze(0).detach())

    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, rewards, dones, next_states, worker_logprobs, next_next_states):     
        action_mean, values             = self.actor(states), self.critic(states)
        old_action_mean, old_values     = self.actor_old(states), self.critic_old(states)
        next_values, next_next_values   = self.critic(next_states), self.critic(next_next_states)

        loss = get_loss(action_mean, old_action_mean, values, old_values, next_values, actions, rewards, dones, worker_logprobs, next_next_values,
                self.action_std, self.policy_kl_range, self.policy_params, self.value_clip, self.vf_loss_coef, self.entropy_coef, self.use_gpu)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss.backward()

        self.actor_optimizer.step() 
        self.critic_optimizer.step() 

    # Update the model
    def update_ppo(self, memory = None): 
        if memory is None:
            memory = self.memory 

        batch_size = 1 if int(len(memory) / self.minibatch) == 0 else int(len(memory) / self.minibatch) 
        dataloader = DataLoader(memory, batch_size, shuffle = False)
        
        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states, worker_logprobs, next_next_states in dataloader:
                self.training_ppo(states.float().to(self.device), actions.float().to(self.device), rewards.float().to(self.device), \
                    dones.float().to(self.device), next_states.float().to(self.device), worker_logprobs.float().to(self.device),\
                    next_next_states.float().to(self.device))

        # Clear the memory
        self.memory.clearMemory()

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

    def get_weights(self):
        return [param.data for name, param in self.actor.named_parameters()]

    def set_weights(self, actor_w):
        for params_cur, params_new in zip(self.actor.named_parameters(), actor_w):
            params_cur[1].data = torch.FloatTensor(params_new).to(set_device(self.use_gpu))

    def log_action(self, states = None, actions = None):
        if states is None and actions is None:
            states, actions, _, _, _, _, _    = self.memory.get_all_items()
        
        states, actions = torch.FloatTensor(states).unsqueeze(0).to(set_device(self.use_gpu)), torch.FloatTensor(actions).to(set_device(self.use_gpu))
        action_mean     = self.actor(states)
        act_logprob     = logprob(action_mean, self.action_std, actions, self.use_gpu)

        return to_numpy(act_logprob.squeeze(0).detach())
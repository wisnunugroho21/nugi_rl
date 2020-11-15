import torch

from distribution.basic import BasicDiscrete, BasicContinous
from agent.agent import Agent
from loss.truly_ppo import TrulyPPO
from utils.pytorch_utils import set_device, to_numpy

class AgentDiscrete(Agent):  
    def __init__(self, Actor_Model, Critic_Model, state_dim, action_dim, 
                is_training_mode = True, policy_kl_range = 0.0008, policy_params = 20, value_clip = 1.0, 
                entropy_coef = 0.0, vf_loss_coef = 1.0, minibatch = 4, PPO_epochs = 4, 
                gamma = 0.99, lam = 0.95, learning_rate = 2.5e-4, folder = 'model', use_gpu = True):
                        
        super(AgentDiscrete, self).__init__(Actor_Model, Critic_Model, state_dim, action_dim, 
                is_training_mode, policy_kl_range, policy_params, value_clip, 
                entropy_coef, vf_loss_coef, minibatch, PPO_epochs, 
                gamma, lam, learning_rate, folder, use_gpu)

        self.distribution   = BasicDiscrete(self.device)
        self.trulyPPO       = TrulyPPO(self.device, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef)      

    def act(self, state):
        state         = torch.FloatTensor(state).unsqueeze(0).to(self.device).detach()
        action_probs  = self.actor(state)
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action = self.distribution.sample(action_probs)
        else:            
            action = torch.argmax(action_probs, 1)
              
        return action

    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, rewards, dones, next_states): 
        action_probs, values          = self.actor(states), self.critic(states)
        old_action_probs, old_values  = self.actor_old(states), self.critic_old(states)
        next_values                   = self.critic(next_states)
        
        loss = self.trulyPPO.get_discrete_loss(action_probs, old_action_probs, values, old_values, next_values, actions, rewards, dones)                

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss.backward()

        self.actor_optimizer.step() 
        self.critic_optimizer.step()

class AgentContinous(Agent):
    def __init__(self, Actor_Model, Critic_Model, state_dim, action_dim, 
                is_training_mode = True, policy_kl_range = 0.03, policy_params = 5, 
                value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0,
                minibatch = 32, PPO_epochs = 10, gamma = 0.99, lam = 0.95, 
                learning_rate = 3e-4, action_std = 1.0, folder = 'model', use_gpu = True): 
        
        super(AgentContinous, self).__init__(Actor_Model, Critic_Model, state_dim, action_dim, 
                is_training_mode, policy_kl_range, policy_params, value_clip, 
                entropy_coef, vf_loss_coef, minibatch, PPO_epochs, 
                gamma, lam, learning_rate, folder, use_gpu)
        
        self.action_std     = action_std
        self.distribution   = BasicContinous(self.device)
        self.trulyPPO       = TrulyPPO(self.device, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef) 

    def set_params(self, params):
        super().set_params(params)
        self.action_std         = self.action_std * params

    def act(self, state):
        state           = torch.FloatTensor(state).unsqueeze(0).to(self.device).detach()
        action_mean     = self.actor(state)
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action = self.distribution.sample(action_mean, self.action_std)
        else:
            action = action_mean  
              
        return to_numpy(action.squeeze(0), self.use_gpu)

    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, rewards, dones, next_states): 
        action_mean, values             = self.actor(states), self.critic(states)
        old_action_mean, old_values     = self.actor_old(states), self.critic_old(states)
        next_values                     = self.critic(next_states)

        loss = self.trulyPPO.get_continous_loss(action_mean, self.action_std, old_action_mean, self.action_std, values, old_values, next_values, actions, rewards, dones)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss.backward()

        self.actor_optimizer.step() 
        self.critic_optimizer.step() 
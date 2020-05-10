import tensorflow as tf

from agent import Agent
from memory.on_policy_impala_memory import OnMemoryImpala

class AgentImpala(Agent):  
    def __init__(self, Actor_Model, Critic_Model, state_dim, action_dim,
                is_training_mode = True, policy_kl_range = 0.0008, policy_params = 20, 
                value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, 
                minibatch = 4, PPO_epochs = 4, gamma = 0.99, 
                lam = 0.95, learning_rate = 2.5e-4, folder = 'model', use_gpu = True):

        super(AgentImpala, self).__init__(Actor_Model, Critic_Model, state_dim, action_dim, 
                is_training_mode, policy_kl_range, policy_params, value_clip, 
                entropy_coef, vf_loss_coef, minibatch, PPO_epochs, 
                gamma, lam, learning_rate, folder, use_gpu) 

    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, rewards, dones, next_states, worker_action_data): 
        pass

    # Update the model
    def update_ppo(self):       
        batch_size = 1 if int(len(self.memory) / self.minibatch) == 0 else int(len(self.memory) / self.minibatch)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states, worker_action_data in self.memory.get_all_items(to_tensor_dataset = True).batch(batch_size): 
                self.training_ppo(states.float().to(self.device), actions.float().to(self.device), \
                    rewards.float().to(self.device), dones.float().to(self.device), next_states.float().to(self.device), \
                    worker_action_data.float().to(self.device))

        # Clear the memory
        self.memory.clearMemory()

        # Copy new weights into old policy:
        self.actor_old.set_weights(self.actor.get_weights())
        self.critic_old.set_weights(self.critic.get_weights())
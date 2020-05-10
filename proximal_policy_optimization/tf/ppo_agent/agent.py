import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from memory.on_policy_memory import OnMemory

class Agent:  
    def __init__(self, Actor_Model, Critic_Model, state_dim, action_dim,
                is_training_mode = True, policy_kl_range = 0.0008, policy_params = 20, 
                value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, 
                minibatch = 4, PPO_epochs = 4, gamma = 0.99, 
                lam = 0.95, learning_rate = 2.5e-4, folder = 'model', use_gpu = True):   

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

        self.critic             = Critic_Model(state_dim, action_dim, use_gpu)
        self.critic_old         = Critic_Model(state_dim, action_dim, use_gpu)
        
        self.optimizer          = Adam(learning_rate = learning_rate)
        self.memory             = OnMemory()
        self.use_gpu            = use_gpu

    def set_params(self, params):
        self.value_clip         = self.value_clip * params if self.value_clip is not None else self.value_clip
        self.policy_kl_range    = self.policy_kl_range * params
        self.policy_params      = self.policy_params * params

    @tf.function
    def act(self, state):
        pass

    # Get loss and Do backpropagation
    @tf.function
    def training_ppo(self, states, actions, rewards, dones, next_states): 
        pass

    # Update the model
    def update_ppo(self):        
        batch_size = 1 if int(len(self.memory) / self.minibatch) == 0 else int(len(self.memory) / self.minibatch)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in self.memory.get_all_items(to_tensor_dataset = True).batch(batch_size): 
                self.training_ppo(states, actions, rewards, dones, next_states)

        # Clear the memory
        self.memory.clearMemory()

        # Copy new weights into old policy:
        self.actor_old.set_weights(self.actor.get_weights())
        self.critic_old.set_weights(self.critic.get_weights())

    def save_weights(self):
        self.actor.save_weights(self.folder + '/actor_ppo', save_format='tf')
        self.actor_old.save_weights(self.folder + '/actor_old_ppo', save_format='tf')
        self.critic.save_weights(self.folder + '/critic_ppo', save_format='tf')
        self.critic_old.save_weights(self.folder + '/critic_old_ppo', save_format='tf')
        
    def load_weights(self):
        self.actor.load_weights(self.folder + '/actor_ppo')
        self.actor_old.load_weights(self.folder + '/actor_old_ppo')
        self.critic.load_weights(self.folder + '/critic_ppo')
        self.critic_old.load_weights(self.folder + '/critic_old_ppo')
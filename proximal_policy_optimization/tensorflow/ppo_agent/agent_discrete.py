import tensorflow as tf
from model.BasicSigmoidNN import Actor_Model, Critic_Model
from memory.on_policy_memory import OnMemory
from ppo_loss.truly_ppo_discrete import get_loss
from distribution.categorical_distribution import sample

class Agent:  
    def __init__(self, action_dim, is_training_mode = True, policy_kl_range = 0.0008, policy_params = 20, value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0,
                 minibatch = 4, PPO_epochs = 4, gamma = 0.99, lam = 0.95, learning_rate = 2.5e-4, folder = 'model'):        
        self.policy_kl_range = policy_kl_range 
        self.policy_params = policy_params
        self.value_clip = value_clip    
        self.entropy_coef = entropy_coef
        self.vf_loss_coef = vf_loss_coef
        self.minibatch = minibatch       
        self.PPO_epochs = PPO_epochs
        self.is_training_mode = is_training_mode
        self.action_dim = action_dim    
        self.learning_rate = learning_rate   
        self.folder = folder       

        self.actor = Actor_Model(action_dim)
        self.actor_old = Actor_Model(action_dim)

        self.critic = Critic_Model(action_dim)
        self.critic_old = Critic_Model(action_dim)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.memory = OnMemory()  

    def set_params(self, params):
        self.value_clip         = self.value_clip * params
        self.policy_kl_range    = self.policy_kl_range * params
        self.policy_params      = self.policy_params * params
        #self.learning_rate      = self.learning_rate * params
        #self.optimizer          = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

    def save_eps(self, state, reward, action, done, next_state):
        self.memory.save_eps(state, reward, action, done, next_state) 

    def save_replace_all_eps(self, states, rewards, actions, dones, next_states):
        self.memory.save_replace_all(states, rewards, actions, dones, next_states)

    def get_eps(self):
        return self.memory.get_all_items() 

    @tf.function
    def act(self, state):      
        state           = tf.expand_dims(tf.cast(state, dtype = tf.float32), 0) 
        action_probs    = self.actor(state)

        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions        
        if self.is_training_mode:
            # Sample the action
            action = sample(action_probs)
        else:
            action = tf.math.argmax(action_probs)

        return action

    # Get loss and Do backpropagation
    @tf.function
    def training_ppo(self, states, actions, rewards, dones, next_states):        
        with tf.GradientTape() as tape:
            action_probs, values         = self.actor(states), self.critic(states)
            old_action_probs, old_values = self.actor_old(states), self.critic_old(states)
            next_values                 = self.critic(next_states)

            loss = get_loss(action_probs, old_action_probs, values, old_values, next_values, actions, rewards, dones,
                self.policy_kl_range, self.policy_params, self.value_clip, self.vf_loss_coef, self.entropy_coef)

        gradients = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)        
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.critic.trainable_variables)) 

    # Update the model
    def update_ppo(self):        
        batch_size = int(self.memory.length() / self.minibatch)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in self.memory.get_dataset_items().batch(batch_size):
                self.training_ppo(states, actions, rewards, dones, next_states)

        # Clear the memory
        self.memory.clearMemory()

        # Copy new weights into old policy:
        self.actor_old.set_weights(self.actor.get_weights())
        self.critic_old.set_weights(self.critic.get_weights())

    def save_weights(self):
        self.actor.save_weights(self.folder + '/actor_ppo', save_format='tf')
        self.critic.save_weights(self.folder + '/critic_ppo', save_format='tf')
        
    def load_weights(self):
        self.actor.load_weights(self.folder + '/actor_ppo')
        self.critic.load_weights(self.folder + '/critic_ppo')

    def get_weights(self):
        return self.actor.get_weights()

    def set_weights(self, actor_w):
        self.actor.set_weights(actor_w)
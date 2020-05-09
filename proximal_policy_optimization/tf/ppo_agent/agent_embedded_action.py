import tensorflow as tf

from distribution.categorical_distribution import sample
from ppo_agent.agent_discrete import AgentDiscrete

class EmbeddedAgentDiscrete(AgentDiscrete):  
    def __init__(self, Actor_Model, Critic_Model, state_dim, action_dim, 
                is_training_mode = True, policy_kl_range = 0.0008, policy_params = 20, value_clip = 1.0, 
                entropy_coef = 0.0, vf_loss_coef = 1.0, minibatch = 4, PPO_epochs = 4, 
                gamma = 0.99, lam = 0.95, learning_rate = 2.5e-4, folder = 'model', use_gpu = True):
                        
        super(EmbeddedAgentDiscrete, self).__init__(Actor_Model, Critic_Model, state_dim, action_dim, 
                is_training_mode, policy_kl_range, policy_params, value_clip, 
                entropy_coef, vf_loss_coef, minibatch, PPO_epochs, 
                gamma, lam, learning_rate, folder, use_gpu)

    @tf.function
    def act(self, state, available_action):
        state               = tf.expand_dims(tf.cast(state, dtype = tf.float32), 0)
        available_action    = tf.expand_dims(tf.cast(available_action, dtype = tf.float64), 0)
        action_probs        = self.actor(state, available_action)
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action = sample(action_probs)
        else:            
            action = tf.math.argmax(action_probs, 1)  
              
        return action
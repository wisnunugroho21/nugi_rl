import tensorflow as tf

from distribution.categorical_distribution import sample, entropy, kl_divergence, logprob
from policy_function.advantage_function import generalized_advantage_estimation
from policy_function.value_function import temporal_difference, monte_carlo_discounted

# Loss for PPO  
def get_loss(action_probs, old_action_probs, values, old_values, next_values, actions, rewards, dones,
            policy_kl_range = 0.0008, policy_params = 20, value_clip = 1.0, vf_loss_coef = 1.0, entropy_coef = 0.01):
    # Don't use old value in backpropagation
    Old_values = tf.stop_gradient(old_values)

    # Getting external general advantages estimator
    Advantages = tf.stop_gradient(generalized_advantage_estimation(values, rewards, next_values, dones))
    Returns = tf.stop_gradient(monte_carlo_discounted(rewards, dones))

    # Finding the ratio (pi_theta / pi_theta__old):        
    logprobs = logprob(action_probs, actions)
    Old_logprobs = tf.stop_gradient(logprob(old_action_probs, actions))

    # Finding Surrogate Loss
    ratios = tf.math.exp(logprobs - Old_logprobs) # ratios = old_logprobs / logprobs        
    Kl = kl_divergence(old_action_probs, action_probs)
    pg_targets = tf.where(
            tf.logical_and(Kl >= policy_kl_range, ratios * Advantages >= 1 * Advantages),
            ratios * Advantages - policy_params * Kl,
            ratios * Advantages
    )
    pg_loss = tf.math.reduce_mean(pg_targets)

    # Getting entropy from the action probability 
    dist_entropy = tf.math.reduce_mean(entropy(action_probs))

    # Getting External critic loss by using Clipped critic value
    vpredclipped = old_values + tf.clip_by_value(values - Old_values, -value_clip, value_clip) # Minimize the difference between old value and new value
    vf_losses1 = tf.math.square(Returns - values) * 0.5 # Mean Squared Error
    vf_losses2 = tf.math.square(Returns - vpredclipped) * 0.5 # Mean Squared Error        
    critic_loss = tf.math.reduce_mean(tf.math.maximum(vf_losses1, vf_losses2))

    #critic_loss = tf.math.square(Returns - values) * 0.5                  

    # We need to maximaze Policy Loss to make agent always find Better Rewards
    # and minimize Critic Loss 
    loss = (critic_loss * vf_loss_coef) - (dist_entropy * entropy_coef) - pg_loss
    return loss
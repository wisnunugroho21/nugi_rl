import torch

from distribution.categorical_distribution import sample, entropy, kldivergence, logprob
from policy_function.advantage_function import generalized_advantage_estimation
from policy_function.value_function import temporal_difference

# Loss for PPO  
def get_loss(action_probs, old_action_probs, values, old_values, next_values, actions, rewards, dones,
            policy_kl_range = 0.0008, policy_params = 20, value_clip = 1.0, vf_loss_coef = 1.0, entropy_coef = 0.01, use_gpu = True):

        # Don't use old value in backpropagation
        Old_values      = old_values.detach()               

        # Getting general advantages estimator and returns
        Advantages      = generalized_advantage_estimation(rewards, values, next_values, dones)
        Returns         = (Advantages + values).detach()
        Advantages      = ((Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)).detach()

        # Finding the ratio (pi_theta / pi_theta__old): 
        logprobs        = logprob(action_probs, actions)
        Old_logprobs    = logprob(old_action_probs, actions).detach()

        # Finding Surrogate Loss
        ratios          = (logprobs - Old_logprobs).exp() # ratios = old_logprobs / logprobs        
        Kl              = kldivergence(old_action_probs, action_probs)

        pg_targets  = torch.where(
                (Kl >= policy_kl_range) & (ratios > 1),
                ratios * Advantages - policy_params * Kl,
                ratios * Advantages
        )
        pg_loss     = pg_targets.mean()

        # Getting Entropy from the action probability 
        dist_entropy = entropy(action_probs, use_gpu).mean()

        # Getting Critic loss by using Clipped critic value
        if value_clip is None:
                critic_loss   = ((Returns - values).pow(2) * 0.5).mean()
        else:
                vpredclipped  = old_values + torch.clamp(values - Old_values, -value_clip, value_clip) # Minimize the difference between old value and new value
                vf_losses1    = (Returns - values).pow(2) * 0.5 # Mean Squared Error
                vf_losses2    = (Returns - vpredclipped).pow(2) * 0.5 # Mean Squared Error        
                critic_loss   = torch.max(vf_losses1, vf_losses2).mean() 

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss 
        loss = (critic_loss * vf_loss_coef) - (dist_entropy * entropy_coef) - pg_loss
        return loss
import torch

from distribution.normal_distribution import sample, entropy, kldivergence, logprob
from policy_function.advantage_function import impala_advantage_estimation
from policy_function.value_function import vtrace

# Loss for PPO  
def get_loss(action_mean, old_action_mean, values, old_values, next_values, actions, rewards, dones, worker_logprobs, next_next_values,
            action_std = 1.0, policy_kl_range = 0.03, policy_params = 5, value_clip = 1.0, vf_loss_coef = 1.0, entropy_coef = 0.0, use_gpu = True, gamma = 0.99):
    
        # Don't use old value in backpropagation
        Old_values  = old_values.detach()

        # Finding the ratio (pi_theta / pi_theta__old):        
        logprobs      = logprob(action_mean, action_std, actions, use_gpu)
        Old_logprobs  = logprob(old_action_mean, action_std, actions, use_gpu).detach()

        # Getting general advantages estimator
        Returns         = vtrace(rewards, values, next_values, dones, worker_logprobs, logprobs).detach()
        NextReturns     = vtrace(rewards, next_values, next_next_values, dones, worker_logprobs, logprobs)      
        Advantages      = impala_advantage_estimation(rewards, values, NextReturns, worker_logprobs, logprobs).detach()

        # Finding Surrogate Loss
        ratios      = (logprobs - Old_logprobs).exp() # ratios = old_logprobs / logprobs        
        Kl          = kldivergence(old_action_mean, action_std, action_mean, action_std, use_gpu)

        pg_targets  = torch.where(
                (Kl >= policy_kl_range) & (ratios >= 1),
                ratios * Advantages - policy_params * Kl,
                ratios * Advantages
        )
        pg_loss     = pg_targets.mean()

        # Getting entropy from the action probability 
        dist_entropy = entropy(action_mean, action_std, use_gpu).mean()

        # Getting External critic loss by using Clipped critic value
        vpredclipped  = old_values + torch.clamp(values - Old_values, -value_clip, value_clip) # Minimize the difference between old value and new value
        vf_losses1    = (Returns - values).pow(2) * 0.5 # Mean Squared Error
        vf_losses2    = (Returns - vpredclipped).pow(2) * 0.5 # Mean Squared Error        
        critic_loss   = torch.max(vf_losses1, vf_losses2).mean()                

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss 
        loss = (critic_loss * vf_loss_coef) - (dist_entropy * entropy_coef) - pg_loss
        return loss
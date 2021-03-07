import torch

from distribution.basic import BasicDiscrete, BasicContinous

class PPOClip():
    def __init__(self, advantage_function, device, policy_clip = 0.2, value_clip = 1.0, vf_loss_coef = 1.0, entropy_coef = 0.01, action_std = 1.0, gamma = 0.99):
        self.policy_clip    = policy_clip
        self.value_clip     = value_clip
        self.vf_loss_coef   = vf_loss_coef
        self.entropy_coef   = entropy_coef
        self.action_std     = action_std

        self.advantagefunction  = advantage_function

        self.discrete    = BasicDiscrete(device)
        self.continous   = BasicContinous(device)

    # Loss for PPO  
    def compute_discrete_loss(self, action_probs, old_action_probs, values, old_values, next_values, actions, rewards, dones):
        # Don't use old value in backpropagation
        Old_values          = old_values.detach()
        Old_action_probs    = old_action_probs.detach()               

        # Getting general advantages estimator and returns
        Advantages      = self.advantagefunction.generalized_advantage_estimation(rewards, values, next_values, dones)
        Returns         = (Advantages + values).detach()
        Advantages      = ((Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)).detach()

        # Finding the ratio (pi_theta / pi_theta__old):
        logprobs        = self.discrete.logprob(action_probs, actions)
        Old_logprobs    = self.discrete.logprob(Old_action_probs, actions).detach()

        # Finding Surrogate Loss
        ratios          = torch.exp(logprobs - Old_logprobs) # ratios = old_logprobs / logprobs
        surr1           = ratios * Advantages
        surr2           = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * Advantages
        pg_loss         = torch.min(surr1, surr2).mean()

        # Getting Entropy from the action probability
        dist_entropy    = self.discrete.entropy(action_probs).mean()

        # Getting Critic loss by using Clipped critic value
        if self.value_clip is None:
            critic_loss   = ((Returns - values).pow(2) * 0.5).mean()
        else:
            vpredclipped  = old_values + torch.clamp(values - Old_values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
            vf_losses1    = (Returns - values).pow(2) * 0.5 # Mean Squared Error
            vf_losses2    = (Returns - vpredclipped).pow(2) * 0.5 # Mean Squared Error        
            critic_loss   = torch.max(vf_losses1, vf_losses2).mean() 

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss 
        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss

    def compute_continous_loss(self, action_mean, action_std, old_action_mean, old_action_std, values, old_values, next_values, actions, rewards, dones):    
        # Don't use old value in backpropagation
        Old_values          = old_values.detach()
        Old_action_mean     = old_action_mean.detach()
        Old_action_std      = old_action_std.detach()          

        # Getting general advantages estimator and returns
        Advantages      = self.advantagefunction.generalized_advantage_estimation(rewards, values, next_values, dones)
        Returns         = (Advantages + values).detach()
        Advantages      = ((Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)).detach() 

        # Finding the ratio (pi_theta / pi_theta__old):
        logprobs        = self.continous.logprob(action_mean, action_std, actions)
        Old_logprobs    = self.continous.logprob(Old_action_mean, Old_action_std, actions).detach() 

        # Finding Surrogate Loss
        ratios          = torch.exp(logprobs - Old_logprobs) # ratios = old_logprobs / logprobs
        surr1           = ratios * Advantages
        surr2           = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * Advantages
        pg_loss         = torch.min(surr1, surr2).mean()

        # Getting entropy from the action probability 
        dist_entropy    = self.continous.entropy(action_mean, action_std).mean()

        # Getting Critic loss by using Clipped critic value
        if self.value_clip is None:
                critic_loss   = ((Returns - values).pow(2) * 0.5).mean()
        else:
                vpredclipped  = old_values + torch.clamp(values - Old_values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
                vf_losses1    = (Returns - values).pow(2) * 0.5 # Mean Squared Error
                vf_losses2    = (Returns - vpredclipped).pow(2) * 0.5 # Mean Squared Error        
                critic_loss   = torch.max(vf_losses1, vf_losses2).mean()                

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss 
        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss
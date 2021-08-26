import torch

class TrulyPPO():
    def __init__(self, distribution, advantage_function, policy_kl_range = 0.0008, policy_params = 20, value_clip = 1.0, vf_loss_coef = 1.0, entropy_coef = 0.01,
        ex_advantages_coef = 2, in_advantages_coef = 1):
        
        self.ex_advantages_coef = ex_advantages_coef
        self.in_advantages_coef = in_advantages_coef

        self.policy_kl_range    = policy_kl_range
        self.policy_params      = policy_params
        self.value_clip         = value_clip
        self.vf_loss_coef       = vf_loss_coef
        self.entropy_coef       = entropy_coef

        self.advantage_function = advantage_function
        self.distribution       = distribution

    def compute_loss(self, action_datas, old_action_datas, ex_values, old_ex_values, next_ex_values, actions, ex_rewards, dones,
        state_preds, state_targets, in_values, old_in_values, next_in_values, std_in_rewards):

        ex_advantages   = self.advantage_function.compute_advantages(ex_rewards, ex_values, next_ex_values, dones).detach()
        ex_returns      = (ex_advantages + ex_values).detach()

        in_rewards      = ((state_targets - state_preds).pow(2) * 0.5 / (std_in_rewards.mean() + 1e-6)).detach()
        in_advantages   = self.advantage_function.compute_advantages(in_rewards, in_values, next_in_values, dones).detach()        
        in_returns      = (in_advantages + in_values).detach()

        advantages      = (self.ex_advantages_coef * ex_advantages + self.in_advantages_coef * in_advantages).detach()

        logprobs        = self.distribution.logprob(action_datas, actions) + 1e-5
        old_logprobs    = (self.distribution.logprob(old_action_datas, actions) + 1e-5).detach()

        ratios          = (logprobs - old_logprobs).exp()       
        Kl              = self.distribution.kldivergence(old_action_datas, action_datas)

        pg_targets      = torch.where(
            (Kl >= self.policy_kl_range) & (ratios > 1.0),
            ratios * advantages - self.policy_params * Kl,
            ratios * advantages
        )
        
        pg_loss         = pg_targets.mean()
        dist_entropy    = self.distribution.entropy(action_datas).mean()

        if self.value_clip is None:
            critic_ex_loss  = ((ex_returns - ex_values).pow(2) * 0.5).mean()
            critic_in_loss  = ((in_returns - in_values).pow(2) * 0.5).mean()
        else:
            ex_vpredclipped = old_ex_values + torch.clamp(ex_values - old_ex_values, -self.value_clip, self.value_clip)
            critic_ex_loss  = ((ex_returns - ex_vpredclipped).pow(2) * 0.5).mean()

            in_vpredclipped = old_in_values + torch.clamp(in_values - old_in_values, -self.value_clip, self.value_clip)
            critic_in_loss  = ((in_returns - in_vpredclipped).pow(2) * 0.5).mean()

        critic_loss = critic_ex_loss + critic_in_loss

        loss = (critic_loss * self.vf_loss_coef) -  (dist_entropy * self.entropy_coef) - pg_loss
        return loss
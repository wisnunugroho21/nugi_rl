import torch
import math

class PhiLoss():
    def __init__(self, distribution, advantage_function):
        self.advantage_function = advantage_function
        self.distribution       = distribution

    def compute_loss(self, action_datas, values, next_values, actions, rewards, dones, temperature):
        temperature         = temperature.detach()

        advantages          = self.advantage_function.compute_advantages(rewards, values, next_values, dones).detach()
        logprobs            = self.distribution.logprob(action_datas, actions)   

        ratio               = advantages / (temperature + 1e-3)
        ratio_max           = (ratio.max(0)[0]).detach()

        psi                 = (ratio - ratio_max).exp() / (ratio - ratio_max).exp().sum()    
        loss                = -1 * (psi * logprobs).sum()

        return loss

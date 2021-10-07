import torch
import math

class PhiLoss():
    def __init__(self, distribution, advantage_function):
        self.advantage_function = advantage_function
        self.distribution       = distribution

    def compute_loss(self, action_datas, values, next_values, actions, rewards, dones, temperature):
        temperature         = temperature.detach()

        advantages          = self.advantage_function.compute_advantages(rewards, values, next_values, dones).detach()
        top_adv, top_idx    = torch.topk(advantages, math.ceil(advantages.size(0) / 2), 0)

        logprobs            = self.distribution.logprob(action_datas, actions)
        top_logprobs        = logprobs[top_idx]        

        ratio               = top_adv / (temperature + 1e-3)
        psi                 = torch.nn.functional.softmax(ratio, dim = 0)

        loss                = -1 * (psi * top_logprobs).sum()
        return loss

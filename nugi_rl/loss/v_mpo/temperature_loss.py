import torch
import math

class TemperatureLoss():
    def __init__(self, advantage_function, coef_temp = 0.0001):
        self.advantage_function = advantage_function
        self.coef_temp          = coef_temp

    def compute_loss(self, values, next_values, rewards, dones, temperature):
        advantages  = self.advantage_function.compute_advantages(rewards, values, next_values, dones).detach()
        top_adv, _  = torch.topk(advantages, math.ceil(len(advantages) / 2), 0)        

        ratio       = top_adv / (temperature + 1e-3)
        ratio_max   = (ratio.max(0)[0]).detach()

        logmeanexp  = ratio_max + (ratio - ratio_max).exp().mean().log()
        loss        = temperature * self.coef_temp + temperature * logmeanexp

        return loss.squeeze()

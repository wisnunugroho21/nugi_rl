import torch
import math

class TemperatureLoss():
    def __init__(self, advantage_function, coef_temp = 0.0001, device = torch.device('cuda')):
        self.advantage_function = advantage_function
        self.coef_temp          = coef_temp
        self.device             = device

    def compute_loss(self, values, next_values, rewards, dones, temperature):
        advantages  = self.advantage_function.compute_advantages(rewards, values, next_values, dones).detach()                
        top_adv, _  = torch.topk(advantages, math.ceil(len(advantages) / 2), 0)

        n           = torch.Tensor([len(top_adv)]).to(self.device)
        ratio       = top_adv / (temperature + 1e-3)

        loss        = temperature * self.coef_temp + temperature * (torch.logsumexp(ratio, dim = 0) - n.log())
        return loss.squeeze()

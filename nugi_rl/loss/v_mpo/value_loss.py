import torch

class ValueLoss():
    def __init__(self, advantage_function, value_clip):
        self.advantage_function = advantage_function
        self.value_clip = value_clip

    def compute_loss(self, values, old_values, next_values, rewards, dones):
        advantages  = self.advantage_function.compute_advantages(rewards, values, next_values, dones)
        returns     = (advantages + values).detach()

        if self.value_clip is None:
            value_loss      = ((returns - values).pow(2) * 0.5).mean()
        else:
            vpredclipped    = old_values + torch.clamp(values - old_values, -self.value_clip, self.value_clip)
            value_loss      = ((returns - vpredclipped).pow(2) * 0.5).mean()

        return value_loss
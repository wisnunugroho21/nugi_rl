import torch

class ValueLoss():
    def __init__(self, advantage_function):
        self.advantage_function = advantage_function

    def compute_loss(self, values, next_values, rewards, dones):
        advantages  = self.advantage_function.compute_advantages(rewards, values, next_values, dones).detach()
        returns     = (advantages + values).detach()

        value_loss  = ((returns - values).pow(2) * 0.5).mean()
        return value_loss
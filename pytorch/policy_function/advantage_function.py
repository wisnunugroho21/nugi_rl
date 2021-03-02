import torch

class AdvantageFunction():
    def __init__(self, gamma = 0.99):
        self.gamma  = gamma

    def generalized_advantage_estimation(self, rewards, values, next_values, dones):
        gae     = 0
        adv     = []     

        delta   = rewards + (1.0 - dones) * self.gamma * next_values - values          
        for step in reversed(range(len(rewards))):  
            gae = delta[step] + (1.0 - dones[step]) * (1.0 - self.gamma) * gae
            adv.insert(0, gae)
            
        return torch.stack(adv)

    def vtrace_advantage_estimation(self, rewards, values, next_values, dones, worker_logprobs, learner_logprobs):
        gae     = 0
        adv     = []

        limit   = torch.FloatTensor([1.0])
        ratio   = torch.min(limit, (learner_logprobs - worker_logprobs).exp())

        delta   = rewards + (1.0 - dones) * self.gamma * next_values - values
        delta   = ratio * delta

        for step in reversed(range(len(rewards))):
            gae   = (1.0 - dones[step]) * (1.0 - self.gamma) * gae
            gae   = delta[step] + ratio * gae
            adv.insert(0, gae)
            
        return torch.stack(adv)
import torch
from utils.pytorch_utils import set_device

def generalized_advantage_estimation(rewards, values, next_values, dones, gamma = 0.99, lam = 0.95):
    gae     = 0
    adv     = []     

    delta   = rewards + (1.0 - dones) * gamma * next_values - values          
    for step in reversed(range(len(rewards))):  
        gae = delta[step] + (1.0 - dones[step]) * gamma * lam * gae
        adv.insert(0, gae)
        
    return torch.stack(adv)

def vtrace_advantage_estimation(rewards, values, next_values, dones, worker_logprobs, learner_logprobs, gamma = 0.99, lam = 0.95):
    gae     = 0
    adv     = []

    limit   = torch.FloatTensor([1.0])
    ratio   = torch.min(limit, (worker_logprobs - learner_logprobs).sum().exp())

    delta   = rewards + (1.0 - dones) * gamma * next_values - values
    delta   = ratio * delta

    for step in reversed(range(len(rewards))):
        gae   = (1.0 - dones[step]) * gamma * lam * gae
        gae   = delta[step] + ratio * gae
        adv.insert(0, gae)
        
    return torch.stack(adv)
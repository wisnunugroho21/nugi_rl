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

    co      = torch.min(1.0, (learner_logprobs - worker_logprobs).exp()).mean(1)
    po      = torch.min(1.0, (learner_logprobs - worker_logprobs).exp()).mean(1)

    delta   = po * (rewards + (1.0 - dones) * gamma * next_values - values)
    for step in reversed(range(len(rewards))):
        gae = delta[step] + (1.0 - dones[step]) * gamma * lam * gae * co[step]
        adv.insert(0, gae)
               
    return torch.stack(adv)

def impala_advantage_estimation(rewards, values, NextReturns, dones, worker_logprobs, learner_logprobs, gamma = 0.99, lam = 0.95):
    po          = torch.min(1.0, (learner_logprobs - worker_logprobs).exp()).mean(1)
    return po * (rewards + (1.0 - dones) * gamma * NextReturns - values)
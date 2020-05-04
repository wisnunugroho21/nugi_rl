import torch
from utils.pytorch_utils import set_device

def generalized_advantage_value_estimation_impala(rewards, values, next_values, dones, worker_logprobs, learner_logprobs, next_next_states, gamma = 0.99, lam = 0.95, use_gpu = True):
    running_add = 0
    v_traces    = []
    adv         = []
    tres        = torch.full_like(worker_logprobs, 1.0)

    co          = torch.min(tres, torch.exp(learner_logprobs - worker_logprobs)).mean(1, keepdim = True)
    po          = torch.min(tres, torch.exp(learner_logprobs - worker_logprobs)).mean(1, keepdim = True)
    delta       = po * (rewards + (1.0 - dones) * gamma * next_values - values)

    for i in reversed(range(len(values))):        
        running_add = values[i] + delta[i] + (1.0 - dones) * gamma * co[i] * (running_add - next_values[i])
        v_traces.insert(0, running_add)        
               
    v_traces    = torch.stack(v_traces).float().to(set_device(use_gpu))
    adv         = torch.stack(adv).float().to(set_device(use_gpu))
    return adv, v_traces
import torch
from utils.pytorch_utils import set_device

def monte_carlo_discounted(reward, done, gamma = 0.99, lam = 0.95):
    returns = []        
    running_add = 0
    
    for i in reversed(range(len(reward))):
        running_add = reward[i] + (1.0 - done) * gamma * running_add  
        returns.insert(0, running_add)
        
    return torch.stack(returns)
    
def temporal_difference(reward, next_value, done, gamma = 0.99, lam = 0.95):
    q_values = reward + (1.0 - done) * gamma * next_value           
    return q_values

# Not Working
def vtrace(rewards, values, next_values, dones, worker_logprobs, learner_logprobs, gamma = 0.99, lam = 0.95):
    running_add = 0
    v_traces    = []

    co          = torch.min(1.0, (learner_logprobs - worker_logprobs).exp()).mean(1)
    po          = torch.min(1.0, (learner_logprobs - worker_logprobs).exp()).mean(1)

    delta       = po * (rewards + (1.0 - dones) * gamma * next_values - values)
    for i in reversed(range(len(values))):        
        #running_add = values[i] + delta[i] + gamma * co[i] * (running_add - next_values[i])
        #running_add = delta[i] + gamma * co[i] * (running_add - next_values[i])
        running_add = delta[i] + (1.0 - dones[i]) * gamma * co[i] * running_add
        v_traces.insert(0, running_add)        
               
    v_traces    = torch.stack(v_traces)
    return v_traces + values
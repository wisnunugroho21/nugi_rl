import torch
from torch import Tensor, device

from nugi_rl.memory.policy.base import PolicyMemory

class PolicyMemory(PolicyMemory):
    def __init__(self, capacity = 1000000, datas = None):
        self.capacity       = capacity

        if datas is None :
            self.states         = torch.tensor([])
            self.actions        = torch.tensor([])
            self.rewards        = torch.tensor([])
            self.dones          = torch.tensor([])
            self.next_states    = torch.tensor([])
            self.logprobs       = torch.tensor([])
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states, self.logprobs = datas

    def __len__(self):
        return self.dones.shape[0]

    def __getitem__(self, idx):
        logprobs = self.logprobs[idx]
        if len(logprobs.shape) == 1:
            logprobs = logprobs.unsqueeze(-1)

        return self.states[idx], self.actions[idx], self.rewards[idx].unsqueeze(-1), self.dones[idx].unsqueeze(-1), self.next_states[idx], logprobs

    def save(self, state: Tensor, action: Tensor, reward: Tensor, done: Tensor, next_state: Tensor, logprob: Tensor) -> None:
        if len(self) >= self.capacity:
            self.states = self.states[1:]
            self.actions = self.actions[1:]
            self.rewards = self.rewards[1:]
            self.dones = self.dones[1:]
            self.next_states = self.next_states[1:]
            self.logprobs = self.logprobs[1:]

        if len(self) == 0:
            self.states         = state.unsqueeze(0)
            self.actions        = action.unsqueeze(0)
            self.rewards        = reward.unsqueeze(0)
            self.dones          = done.unsqueeze(0)
            self.next_states    = next_state.unsqueeze(0)
            self.logprobs       = logprob.unsqueeze(0)

        else:
            self.states         = torch.cat((self.states, state.unsqueeze(0)), dim = 0)
            self.actions        = torch.cat((self.actions, action.unsqueeze(0)), dim = 0)
            self.rewards        = torch.cat((self.rewards, reward.unsqueeze(0)), dim = 0)
            self.dones          = torch.cat((self.dones, done.unsqueeze(0)), dim = 0)
            self.next_states    = torch.cat((self.next_states, next_state.unsqueeze(0)), dim = 0)
            self.logprobs       = torch.cat((self.logprobs, logprob.unsqueeze(0)), dim = 0)

    def get(self, start_position: int = 0, end_position: int = None):
        if end_position is not None and end_position != -1:
            states      = self.states[start_position : end_position + 1]
            actions     = self.actions[start_position : end_position + 1]
            rewards     = self.rewards[start_position : end_position + 1]
            dones       = self.dones[start_position : end_position + 1]
            next_states = self.next_states[start_position : end_position + 1]
            logprobs    = self.logprobs[start_position : end_position + 1]

        else:
            states      = self.states[start_position :]
            actions     = self.actions[start_position :]
            rewards     = self.rewards[start_position :]
            dones       = self.dones[start_position :]
            next_states = self.next_states[start_position :]
            logprobs    = self.logprobs[start_position :]

        return states, actions, rewards, dones, next_states, logprobs

    def clear(self, start_position: int = 0, end_position: int = None) -> None:
        if start_position is not None and start_position > 0 and end_position is not None and end_position != -1:
            self.states         = torch.cat([self.states[ : start_position], self.states[end_position + 1 : ]])
            self.actions        = torch.cat([self.actions[ : start_position], self.actions[end_position + 1 : ]])
            self.rewards        = torch.cat([self.rewards[ : start_position], self.rewards[end_position + 1 : ]])
            self.dones          = torch.cat([self.dones[ : start_position], self.dones[end_position + 1 : ]])
            self.next_states    = torch.cat([self.next_states[ : start_position], self.next_states[end_position + 1 : ]])
            self.logprobs       = torch.cat([self.logprobs[ : start_position], self.logprobs[end_position + 1 : ]])

        elif start_position is not None and start_position > 0:
            self.states         = self.states[ : start_position]
            self.actions        = self.actions[ : start_position]
            self.rewards        = self.rewards[ : start_position]
            self.dones          = self.dones[ : start_position]
            self.next_states    = self.next_states[ : start_position]
            self.logprobs       = self.logprobs[ : start_position]

        elif end_position is not None and end_position != -1:
            self.states         = self.states[end_position + 1 : ]
            self.actions        = self.actions[end_position + 1 : ]
            self.rewards        = self.rewards[end_position + 1 : ]
            self.dones          = self.dones[end_position + 1 : ]
            self.next_states    = self.next_states[end_position + 1 : ]
            self.logprobs       = self.logprobs[end_position + 1 : ]
            
        else:
            del self.states
            del self.actions
            del self.rewards
            del self.dones
            del self.next_states
            del self.logprobs

            self.states         = torch.tensor([])
            self.actions        = torch.tensor([])
            self.rewards        = torch.tensor([])
            self.dones          = torch.tensor([])
            self.next_states    = torch.tensor([])
            self.logprobs       = torch.tensor([])
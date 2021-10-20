import torch
from torch import Tensor

from nugi_rl.memory.policy.base import Memory

class TensorMemory(Memory):
    def __init__(self, device, capacity = 100000, datas = None):
        self.capacity       = capacity
        self.device         = device

        if datas is None :
            self.states         = torch.tensor([]).to(device)
            self.actions        = torch.tensor([]).to(device)
            self.rewards        = torch.tensor([]).to(device)
            self.dones          = torch.tensor([]).to(device)
            self.next_states    = torch.tensor([]).to(device)
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states = datas

    def __len__(self):
        return self.dones.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx].unsqueeze(-1), self.dones[idx].unsqueeze(-1), self.next_states[idx]

    def save(self, state: Tensor, action: Tensor, reward: Tensor, done: Tensor, next_state: Tensor) -> None:
        if len(self) >= self.capacity:
            self.states = self.states[1:]
            self.actions = self.actions[1:]
            self.rewards = self.rewards[1:]
            self.dones = self.dones[1:]
            self.next_states = self.next_states[1:]

        if len(self) == 0:
            self.states         = state.unsqueeze(0)
            self.actions        = action.unsqueeze(0)
            self.rewards        = reward.unsqueeze(0)
            self.dones          = done.unsqueeze(0)
            self.next_states    = next_state.unsqueeze(0)
        else:
            self.states         = torch.cat((self.states, state.unsqueeze(0)), dim = 0)
            self.actions        = torch.cat((self.actions, action.unsqueeze(0)), dim = 0)
            self.rewards        = torch.cat((self.rewards, reward.unsqueeze(0)), dim = 0)
            self.dones          = torch.cat((self.dones, done.unsqueeze(0)), dim = 0)
            self.next_states    = torch.cat((self.next_states, next_state.unsqueeze(0)), dim = 0)

    def get(self, start_position: int = 0, end_position: int = None):
        if end_position is not None and end_position != -1:
            states      = self.states[start_position : end_position + 1]
            actions     = self.actions[start_position : end_position + 1]
            rewards     = self.rewards[start_position : end_position + 1]
            dones       = self.dones[start_position : end_position + 1]
            next_states = self.next_states[start_position : end_position + 1]
        else:
            states      = self.states[start_position :]
            actions     = self.actions[start_position :]
            rewards     = self.rewards[start_position :]
            dones       = self.dones[start_position :]
            next_states = self.next_states[start_position :]

        return states, actions, rewards, dones, next_states

    def clear(self, start_position: int = 0, end_position: int = None) -> None:
        if start_position is not None and start_position > 0:
            self.states         = torch.cat([self.states[0 : start_position], self.states[end_position + 1 : ]])
            self.actions        = torch.cat([self.actions[0 : start_position], self.actions[end_position + 1 : ]])
            self.rewards        = torch.cat([self.rewards[0 : start_position], self.rewards[end_position + 1 : ]])
            self.dones          = torch.cat([self.dones[0 : start_position], self.dones[end_position + 1 : ]])
            self.next_states    = torch.cat([self.next_states[0 : start_position], self.next_states[end_position + 1 : ]])
        elif end_position is not None:
            self.states         = self.states[end_position + 1 : ]
            self.actions        = self.actions[end_position + 1 : ]
            self.rewards        = self.rewards[end_position + 1 : ]
            self.dones          = self.dones[end_position + 1 : ]
            self.next_states    = self.next_states[end_position + 1 : ]
        else:
            del self.states
            del self.actions
            del self.rewards
            del self.dones
            del self.next_states

            self.states         = torch.tensor([]).to(self.device)
            self.actions        = torch.tensor([]).to(self.device)
            self.rewards        = torch.tensor([]).to(self.device)
            self.dones          = torch.tensor([]).to(self.device)
            self.next_states    = torch.tensor([]).to(self.device)

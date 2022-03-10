import torch
from torch import Tensor

from nugi_rl.memory.base import Memory

class PPGMemory(Memory):
    def __init__(self, capacity = 1000000):
        self.capacity       = capacity
        self.states         = torch.tensor([])

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        if isinstance(self.states, list):
            states  = [s[idx] for s in self.states]                
        else:
            states  = self.states[idx]

        return states

    def save(self, state: Tensor) -> None:
        if len(self) >= self.capacity:
            self.states = self.states[1:]

            if isinstance(state, list):
                self.states = [s[1:] for s in self.states]
            else:
                self.states = self.states[1:]

        if len(self) == 0:
            if isinstance(state, list):
                self.states = [s.unsqueeze(0) for s in self.states]
            else:
                self.states = state.unsqueeze(0)
            
        else:
            if isinstance(state, list):
                self.states = [torch.cat((ss,  s.unsqueeze(0)), dim = 0) for ss, s in zip(self.states, state)]
            else:
                self.states = torch.cat((self.states, state.unsqueeze(0)), dim = 0)

    def save_all(self, states: Tensor) -> None:
        for state in zip(states):
            self.save(state)

    def get(self, start_position: int = 0, end_position: int = None):
        if end_position is not None and end_position != -1:
            if isinstance(self.states, list):
                states  = [s[start_position : end_position + 1] for s in self.states]
            else:
                states  = self.states[start_position : end_position + 1]

        else:
            if isinstance(self.states, list):
                states  = [s[start_position :] for s in self.states]
            else:
                states  = self.states[start_position :]

        return states

    def clear(self, start_position: int = 0, end_position: int = None) -> None:
        if start_position is not None and start_position > 0 and end_position is not None and end_position != -1:
            if isinstance(self.states, list):
                self.states         = [torch.cat([s[ : start_position], s[end_position + 1 : ]]) for s in self.states]
            else:
                self.states         = torch.cat([self.states[ : start_position], self.states[end_position + 1 : ]])

        elif start_position is not None and start_position > 0:
            if isinstance(self.states, list):
                self.states         = [s[ : start_position] for s in self.states]
            else:
                self.states         = self.states[ : start_position]

        elif end_position is not None and end_position != -1:
            if isinstance(self.states, list):
                self.states         = [s[end_position + 1 : ] for s in self.states]
            else:
                self.states         = self.states[end_position + 1 : ]
            
        else:
            if isinstance(self.states, list):
                ln_s = len(self.states)
                del self.states
                self.states = [torch.tensor([]) for _ in range(ln_s)]
            else:
                del self.states
                self.states = torch.tensor([])

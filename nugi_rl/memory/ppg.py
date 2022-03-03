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
        return self.states[idx]

    def save(self, state: Tensor) -> None:
        if len(self) >= self.capacity:
            self.states = self.states[1:]

            if isinstance(state, list):
                for i in range(len(self.states)):
                    self.states = self.states[i][1:]
            else:
                self.states = self.states[1:]

        if len(self) == 0:
            if isinstance(state, list):
                self.states         = []
                for i in range(len(state)):                    
                    self.states.append(state[i].unsqueeze(0))
            else:
                self.states = state.unsqueeze(0)
            
        else:
            if isinstance(state, list):
                for i in range(len(state)):
                    self.states[i]  = torch.cat((self.states[i],  state[i].unsqueeze(0)), dim = 0)
            else:
                self.states = torch.cat((self.states, state.unsqueeze(0)), dim = 0)

    def save_all(self, states: Tensor) -> None:
        for state in zip(states):
            self.save(state)

    def get(self, start_position: int = 0, end_position: int = None):
        if end_position is not None and end_position != -1:
            states  = self.states[start_position : end_position + 1]

        else:
            states  = self.states[start_position :]

        return states

    def clear(self, start_position: int = 0, end_position: int = None) -> None:
        if start_position is not None and start_position > 0 and end_position is not None and end_position != -1:
            if isinstance(self.states, list):
                for i in range(len(self.states)):
                    self.states[i]  = torch.cat([self.states[i][ : start_position], self.states[i][end_position + 1 : ]])
            else:
                self.states = torch.cat([self.states[ : start_position], self.states[end_position + 1 : ]])
        
        elif start_position is not None and start_position > 0:
            if isinstance(self.states, list):
                for i in range(len(self.states)):
                    self.states[i] = self.states[i][ : start_position]
            else:
                self.states = self.states[ : start_position]
        
        elif end_position is not None and end_position != -1:
            if isinstance(self.states, list):
                for i in range(len(self.states)):
                    self.states[i] = self.states[i][end_position + 1 : ]
            else:
                self.states = self.states[end_position + 1 : ]
            
        else:
            if isinstance(self.states, list):
                ln_s    = len(self.states)
                del self.states
                self.states = []

                for i in range(ln_s):
                    self.states.append(torch.tensor([]))
            else:
                del self.states
                self.states = torch.tensor([])
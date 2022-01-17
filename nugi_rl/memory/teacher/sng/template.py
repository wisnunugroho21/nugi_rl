import torch
from torch import Tensor

from nugi_rl.memory.base import Memory

class SNGTemplateMemory(Memory):
    def __init__(self, capacity = 1000000):
        self.capacity       = capacity

        self.states         = torch.tensor([])
        self.goals          = torch.tensor([])
        self.next_states    = torch.tensor([])

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.goals[idx], self.next_states[idx]

    def save(self, state: Tensor, goal: Tensor, next_state: Tensor) -> None:
        if len(self) >= self.capacity:
            self.states         = self.states[1:]
            self.goals          = self.goals[1:]
            self.next_states    = self.next_states[1:]

        if len(self) == 0:
            self.states         = state.unsqueeze(0)
            self.goals          = goal.unsqueeze(0)
            self.next_states    = next_state.unsqueeze(0)

        else:
            self.states         = torch.cat((self.states, state.unsqueeze(0)), dim = 0)
            self.goals          = torch.cat((self.goals, goal.unsqueeze(0)), dim = 0)
            self.next_states    = torch.cat((self.next_states, next_state.unsqueeze(0)), dim = 0)

    def save_all(self, states: Tensor, goals: Tensor, next_states: Tensor) -> None:
        for state, goal, next_state in zip(states, goals, next_states):
            self.save(state, goal, next_state)

    def get(self, start_position: int = 0, end_position: int = None):
        if end_position is not None and end_position != -1:
            states      = self.states[start_position : end_position + 1]
            goals       = self.goals[start_position : end_position + 1]
            next_states = self.next_states[start_position : end_position + 1]

        else:
            states      = self.states[start_position :]
            goals       = self.goals[start_position :]
            next_states = self.next_states[start_position :]

        return states, goals, next_states

    def clear(self, start_position: int = 0, end_position: int = None) -> None:
        if start_position is not None and start_position > 0 and end_position is not None and end_position != -1:
            self.states         = torch.cat([self.states[ : start_position], self.states[end_position + 1 : ]])
            self.goals          = torch.cat([self.goals[ : start_position], self.goals[end_position + 1 : ]])
            self.next_states    = torch.cat([self.next_states[ : start_position], self.next_states[end_position + 1 : ]])

        elif start_position is not None and start_position > 0:
            self.states         = self.states[ : start_position]
            self.goals          = self.goals[ : start_position]
            self.next_states    = self.next_states[ : start_position]

        elif end_position is not None and end_position != -1:
            self.states         = self.states[end_position + 1 : ]
            self.goals          = self.goals[end_position + 1 : ]
            self.next_states    = self.next_states[end_position + 1 : ]
            
        else:
            del self.states
            del self.goals
            del self.next_states

            self.states         = torch.tensor([])
            self.goals          = torch.tensor([])
            self.next_states    = torch.tensor([])
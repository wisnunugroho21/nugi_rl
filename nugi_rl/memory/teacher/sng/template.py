import torch

from torch import Tensor
from typing import List, Tuple, Union

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
        if isinstance(self.states, list):
            states      = [s[idx] for s in self.states]
            next_states = [ns[idx] for ns in self.next_states]
                
        else:
            states      = self.states[idx]
            next_states = self.next_states[idx]

        return states, self.goals[idx], next_states

    def save(self, state: Union[Tensor, List[Tensor]], goal: Tensor, next_state: Union[Tensor, List[Tensor]]) -> None:
        if len(self) >= self.capacity:
            self.goals  = self.goals[1:]

            if isinstance(state, list):
                self.states         = [s[1:] for s in self.states]
                self.next_states    = [ns[1:] for ns in self.next_states]
            else:
                self.states         = self.states[1:]
                self.next_states    = self.next_states[1:]

        if len(self) == 0:
            self.goals  = goal.unsqueeze(0)

            if isinstance(state, list):
                self.states         = [s[i].unsqueeze(0) for s in self.states]
                self.next_states    = [ns[i].unsqueeze(0) for ns in self.next_states]
            else:
                self.states         = state.unsqueeze(0)
                self.next_states    = next_state.unsqueeze(0)

        else:
            self.goals  = torch.cat((self.goals, goal.unsqueeze(0)), dim = 0)

            if isinstance(state, list):
                self.states = [torch.cat((ss,  s.unsqueeze(0)), dim = 0) for ss, s in zip(self.states, state)]
                self.states = [torch.cat((nss,  ns.unsqueeze(0)), dim = 0) for nss, ns in zip(self.next_states, next_state)]
            else:
                self.states         = torch.cat((self.states, state.unsqueeze(0)), dim = 0)
                self.next_states    = torch.cat((self.next_states, next_state.unsqueeze(0)), dim = 0)

    def save_all(self, states: Union[Tensor, List[Tensor]], goals: Tensor, next_states: Union[Tensor, List[Tensor]]) -> None:
        for state, goal, next_state in zip(states, goals, next_states):
            self.save(state, goal, next_state)

    def get(self, start_position: int = 0, end_position: int = None):
        if end_position is not None and end_position != -1:
            goals = self.goals[start_position : end_position + 1]

            if isinstance(self.states, list):
                states      = [s[start_position : end_position + 1] for s in self.states]
                next_states = [ns[start_position : end_position + 1] for ns in self.next_states]
            else:
                states      = self.states[start_position : end_position + 1]
                next_states = self.next_states[start_position : end_position + 1]

        else:
            goals = self.goals[start_position :]

            if isinstance(self.states, list):
                states      = [s[start_position :] for s in self.states]
                next_states = [ns[start_position :] for ns in self.next_states]
            else:
                states      = self.states[start_position :]
                next_states = self.next_states[start_position :]

        return states, goals, next_states

    def clear(self, start_position: int = 0, end_position: int = None) -> None:
        if start_position is not None and start_position > 0 and end_position is not None and end_position != -1:
            self.goals = torch.cat([self.goals[ : start_position], self.goals[end_position + 1 : ]])

            if isinstance(self.states, list):
                self.states         = [torch.cat([s[ : start_position], s[end_position + 1 : ]]) for s in self.states]
                self.next_states    = [torch.cat([ns[ : start_position], ns[end_position + 1 : ]]) for ns in self.next_states]
            else:
                self.states         = torch.cat([self.states[ : start_position], self.states[end_position + 1 : ]])
                self.next_states    = torch.cat([self.next_states[ : start_position], self.next_states[end_position + 1 : ]])

        elif start_position is not None and start_position > 0:
            self.goals = self.goals[ : start_position]

            if isinstance(self.states, list):
                self.states         = [s[ : start_position] for s in self.states]
                self.next_states    = [ns[ : start_position] for ns in self.next_states]
            else:
                self.states         = self.states[ : start_position]
                self.next_states    = self.next_states[ : start_position]

        elif end_position is not None and end_position != -1:
            self.goals = self.goals[end_position + 1 : ]

            if isinstance(self.states, list):
                self.states         = [s[end_position + 1 : ] for s in self.states]
                self.next_states    = [ns[end_position + 1 : ] for ns in self.next_states]
            else:
                self.states         = self.states[end_position + 1 : ]
                self.next_states    = self.next_states[end_position + 1 : ]
            
        else:
            if isinstance(self.states, list):
                ln_s    = len(self.states)
                ln_ns   = len(self.next_states)

                del self.states
                del self.next_states

                self.states         = [torch.tensor([]) for _ in range(ln_s)]
                self.next_states    = [torch.tensor([]) for _ in range(ln_ns)]
            else:
                del self.states
                del self.next_states

                self.states         = torch.tensor([])
                self.next_states    = torch.tensor([])

            del self.goals
            self.goals          = torch.tensor([])
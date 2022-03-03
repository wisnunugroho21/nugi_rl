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
            states  = []
            for s in self.states:
                states.append(s[idx])

            next_states = []
            for ns in self.next_states:
                next_states.append(ns[idx])
                
        else:
            states = self.states[idx]
            next_states = self.next_states[idx]

        return states, self.goals[idx], next_states

    def save(self, state: Union[Tensor, List[Tensor]], goal: Tensor, next_state: Union[Tensor, List[Tensor]]) -> None:
        if len(self) >= self.capacity:
            self.goals  = self.goals[1:]

            if isinstance(state, list):
                for i in range(len(self.states)):
                    self.states      = self.states[i][1:]
                    self.next_states = self.next_states[i][1:]
            else:
                self.states         = self.states[1:]
                self.next_states    = self.next_states[1:]

        if len(self) == 0:
            self.goals  = goal.unsqueeze(0)

            if isinstance(state, list):
                self.states         = []
                self.next_states    = []

                for i in range(len(state)):                    
                    self.states.append(state[i].unsqueeze(0))
                    self.next_states.append(next_state[i].unsqueeze(0))
            else:
                self.states         = state.unsqueeze(0)
                self.next_states    = next_state.unsqueeze(0)

        else:
            self.goals  = torch.cat((self.goals, goal.unsqueeze(0)), dim = 0)

            if isinstance(state, list):
                for i in range(len(state)):
                    self.states[i]      = torch.cat((self.states[i],  state[i].unsqueeze(0)), dim = 0)
                    self.next_states[i] = torch.cat((self.next_states[i], next_state[i].unsqueeze(0)), dim = 0)
            else:
                self.states         = torch.cat((self.states, state.unsqueeze(0)), dim = 0)
                self.next_states    = torch.cat((self.next_states, next_state.unsqueeze(0)), dim = 0)

    def save_all(self, states: Tensor, goals: Tensor, next_states: Tensor) -> None:
        for state, goal, next_state in zip(states, goals, next_states):
            self.save(state, goal, next_state)

    def get(self, start_position: int = 0, end_position: int = None):
        if end_position is not None and end_position != -1:
            goals = self.goals[start_position : end_position + 1]

            if isinstance(self.states, list):
                states      = self.states
                next_states = self.next_states

                for i in range(len(states)):
                    states[i] = states[i][start_position : end_position + 1]
                
                for i in range(len(next_states)):
                    next_states[i] = next_states[i][start_position : end_position + 1]
            else:
                states      = self.states[start_position : end_position + 1]
                next_states = self.next_states[start_position : end_position + 1]

        else:
            goals = self.goals[start_position :]

            if isinstance(self.states, list):
                states  = self.states
                next_states = self.next_states

                for i in range(len(states)):
                    states[i] = states[i][start_position :]

                for i in range(len(next_states)):
                    next_states[i] = next_states[i][start_position :]
            else:
                states      = self.states[start_position :]
                next_states = self.next_states[start_position :]

        return states, goals, next_states

    def clear(self, start_position: int = 0, end_position: int = None) -> None:
        if start_position is not None and start_position > 0 and end_position is not None and end_position != -1:
            self.goals = torch.cat([self.goals[ : start_position], self.goals[end_position + 1 : ]])

            if isinstance(self.states, list):
                for i in range(len(self.states)):
                    self.states[i]  = torch.cat([self.states[i][ : start_position], self.states[i][end_position + 1 : ]])
                
                for i in range(len(self.next_states)):
                    self.next_states[i] = torch.cat([self.next_states[i][ : start_position], self.next_states[end_position + 1 : ]])
            else:
                self.states         = torch.cat([self.states[ : start_position], self.states[end_position + 1 : ]])
                self.next_states    = torch.cat([self.next_states[ : start_position], self.next_states[end_position + 1 : ]])

        elif start_position is not None and start_position > 0:
            self.goals = self.goals[ : start_position]

            if isinstance(self.states, list):
                for i in range(len(self.states)):
                    self.states[i] = self.states[i][ : start_position]
                
                for i in range(len(self.next_states)):
                    self.next_states[i] = self.next_states[i][ : start_position]
            else:
                self.states         = self.states[ : start_position]
                self.next_states    = self.next_states[ : start_position]

        elif end_position is not None and end_position != -1:
            self.goals = self.goals[end_position + 1 : ]

            if isinstance(self.states, list):
                for i in range(len(self.states)):
                    self.states[i] = self.states[i][end_position + 1 : ]
                
                for i in range(len(self.next_states)):
                    self.next_states[i] = self.next_states[i][end_position + 1 : ]
            else:
                self.states         = self.states[end_position + 1 : ]
                self.next_states    = self.next_states[end_position + 1 : ]
            
        else:
            if isinstance(self.states, list):
                ln_s    = len(self.states)
                ln_ns   = len(self.next_states)

                del self.states
                del self.next_states

                self.states         = []
                self.next_states    = []

                for i in range(ln_s):
                    self.states.append(torch.tensor([]))
                
                for i in range(ln_ns):
                    self.next_states.append(torch.tensor([]))
            else:
                del self.states
                del self.next_states

                self.states         = torch.tensor([])
                self.next_states    = torch.tensor([])

            del self.goals
            self.goals          = torch.tensor([])
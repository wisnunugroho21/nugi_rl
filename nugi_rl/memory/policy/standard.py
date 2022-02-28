import torch

from torch import Tensor
from typing import List, Tuple, Union

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
        actions = self.actions[idx]
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(-1)

        logprobs = self.logprobs[idx]
        if len(logprobs.shape) == 1:
            logprobs = logprobs.unsqueeze(-1)

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

        return states, actions, self.rewards[idx].unsqueeze(-1), self.dones[idx].unsqueeze(-1), next_states, logprobs

    def save(self, state: Union[Tensor, List[Tensor]], action: Tensor, reward: Tensor, done: Tensor, next_state: Tensor, logprob: Tensor) -> None:
        if len(self) >= self.capacity:
            self.actions        = self.actions[1:]
            self.rewards        = self.rewards[1:]
            self.dones          = self.dones[1:]
            self.logprobs       = self.logprobs[1:]

            if isinstance(state, list):
                for i in range(len(self.states)):
                    self.states      = self.states[i][1:]
                    self.next_states = self.next_states[i][1:]
            else:
                self.states         = self.states[1:]
                self.next_states    = self.next_states[1:]

            torch.cat()

        if len(self) == 0:            
            self.actions        = action.unsqueeze(0)
            self.rewards        = reward.unsqueeze(0)
            self.dones          = done.unsqueeze(0)            
            self.logprobs       = logprob.unsqueeze(0)

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
            self.actions        = torch.cat((self.actions, action.unsqueeze(0)), dim = 0)
            self.rewards        = torch.cat((self.rewards, reward.unsqueeze(0)), dim = 0)
            self.dones          = torch.cat((self.dones, done.unsqueeze(0)), dim = 0)
            self.logprobs       = torch.cat((self.logprobs, logprob.unsqueeze(0)), dim = 0)

            if isinstance(state, list):
                for i in range(len(state)):
                    self.states[i]      = torch.cat((self.states[i],  state[i].unsqueeze(0)), dim = 0)
                    self.next_states[i] = torch.cat((self.next_states[i], next_state[i].unsqueeze(0)), dim = 0)
            else:
                self.states         = torch.cat((self.states, state.unsqueeze(0)), dim = 0)
                self.next_states    = torch.cat((self.next_states, next_state.unsqueeze(0)), dim = 0)

    def get(self, start_position: int = 0, end_position: int = None) -> Tuple[Union[Tensor, List[Tensor]], Tensor, Tensor, Tensor, Union[Tensor, List[Tensor]], Tensor]:
        if end_position is not None and end_position != -1:
            actions     = self.actions[start_position : end_position + 1]
            rewards     = self.rewards[start_position : end_position + 1]
            dones       = self.dones[start_position : end_position + 1]
            logprobs    = self.logprobs[start_position : end_position + 1]

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
            states      = self.states[start_position :]
            actions     = self.actions[start_position :]
            rewards     = self.rewards[start_position :]
            dones       = self.dones[start_position :]
            next_states = self.next_states[start_position :]
            logprobs    = self.logprobs[start_position :]

            if isinstance(self.states, list):
                states  = self.states
                next_states = self.next_states

                for i in range(len(self.states)):
                    states[i] = states[i][start_position :]

                for i in range(len(self.next_states)):
                    next_states[i] = next_states[i][start_position :]
            else:
                states      = self.states[start_position :]
                next_states = self.next_states[start_position :]

        return states, actions, rewards, dones, next_states, logprobs

    def clear(self, start_position: int = 0, end_position: int = None) -> None:
        if start_position is not None and start_position > 0 and end_position is not None and end_position != -1:            
            self.actions        = torch.cat([self.actions[ : start_position], self.actions[end_position + 1 : ]])
            self.rewards        = torch.cat([self.rewards[ : start_position], self.rewards[end_position + 1 : ]])
            self.dones          = torch.cat([self.dones[ : start_position], self.dones[end_position + 1 : ]])            
            self.logprobs       = torch.cat([self.logprobs[ : start_position], self.logprobs[end_position + 1 : ]])

            if isinstance(self.states, list):
                for i in range(len(self.states)):
                    self.states[i]  = torch.cat([self.states[i][ : start_position], self.states[i][end_position + 1 : ]])
                
                for i in range(len(self.next_states)):
                    self.next_states[i] = torch.cat([self.next_states[i][ : start_position], self.next_states[end_position + 1 : ]])
            else:
                self.states         = torch.cat([self.states[ : start_position], self.states[end_position + 1 : ]])
                self.next_states    = torch.cat([self.next_states[ : start_position], self.next_states[end_position + 1 : ]])

        elif start_position is not None and start_position > 0:            
            self.actions    = self.actions[ : start_position]
            self.rewards    = self.rewards[ : start_position]
            self.dones      = self.dones[ : start_position]            
            self.logprobs   = self.logprobs[ : start_position]

            if isinstance(self.states, list):
                for i in range(len(self.states)):
                    self.states[i] = self.states[i][ : start_position]
                
                for i in range(len(self.next_states)):
                    self.next_states[i] = self.next_states[i][ : start_position]
            else:
                self.states         = self.states[ : start_position]
                self.next_states    = self.next_states[ : start_position]

        elif end_position is not None and end_position != -1:
            self.actions        = self.actions[end_position + 1 : ]
            self.rewards        = self.rewards[end_position + 1 : ]
            self.dones          = self.dones[end_position + 1 : ]
            self.logprobs       = self.logprobs[end_position + 1 : ]

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

            del self.actions
            del self.rewards
            del self.dones
            del self.logprobs

            self.actions        = torch.tensor([])
            self.rewards        = torch.tensor([])
            self.dones          = torch.tensor([])
            self.logprobs       = torch.tensor([])            
import torch

from torch import Tensor
from typing import List, Tuple, Union

from nugi_rl.memory.policy.base import PolicyMemory

class PolicyMemory(PolicyMemory):
    def __init__(self, capacity = 1000000, datas = None):
        self.capacity       = capacity

        if datas is None:
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
        if isinstance(self.states, list):
            states      = [s[idx] for s in self.states]
            next_states = [ns[idx] for ns in self.next_states]
                
        else:
            states      = self.states[idx]
            next_states = self.next_states[idx]

        return states, self.actions[idx], self.rewards[idx].unsqueeze(-1), self.dones[idx].unsqueeze(-1), next_states, self.logprobs[idx]

    def save(self, state: Union[Tensor, List[Tensor]], action: Tensor, reward: Tensor, done: Tensor, next_state: Union[Tensor, List[Tensor]], logprob: Tensor) -> None:
        if len(self) >= self.capacity:
            self.actions        = self.actions[1:]
            self.rewards        = self.rewards[1:]
            self.dones          = self.dones[1:]
            self.logprobs       = self.logprobs[1:]

            if isinstance(state, list):
                self.states      = [s[1:] for s in self.states]
                self.next_states = [ns[1:] for ns in self.next_states]
            else:
                self.states         = self.states[1:]
                self.next_states    = self.next_states[1:]

        if len(self) == 0:            
            self.actions        = action.unsqueeze(0)
            self.rewards        = reward.unsqueeze(0)
            self.dones          = done.unsqueeze(0)            
            self.logprobs       = logprob.unsqueeze(0)

            if isinstance(state, list):
                self.states         = [s.unsqueeze(0) for s in self.states]
                self.next_states    = [ns.unsqueeze(0) for ns in self.next_states]
            else:
                self.states         = state.unsqueeze(0)
                self.next_states    = next_state.unsqueeze(0)

        else:
            self.actions        = torch.cat((self.actions, action.unsqueeze(0)), dim = 0)
            self.rewards        = torch.cat((self.rewards, reward.unsqueeze(0)), dim = 0)
            self.dones          = torch.cat((self.dones, done.unsqueeze(0)), dim = 0)
            self.logprobs       = torch.cat((self.logprobs, logprob.unsqueeze(0)), dim = 0)

            if isinstance(state, list):
                self.states = [torch.cat((ss,  s.unsqueeze(0)), dim = 0) for ss, s in zip(self.states, state)]
                self.states = [torch.cat((nss,  ns.unsqueeze(0)), dim = 0) for nss, ns in zip(self.next_states, next_state)]
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
                states      = [s[start_position : end_position + 1] for s in self.states]
                next_states = [ns[start_position : end_position + 1] for ns in self.next_states]
            else:
                states      = self.states[start_position : end_position + 1]
                next_states = self.next_states[start_position : end_position + 1]

        else:
            actions     = self.actions[start_position :]
            rewards     = self.rewards[start_position :]
            dones       = self.dones[start_position :]
            logprobs    = self.logprobs[start_position :]

            if isinstance(self.states, list):
                states      = [s[start_position :] for s in self.states]
                next_states = [ns[start_position :] for ns in self.next_states]
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
                self.states         = [torch.cat([s[ : start_position], s[end_position + 1 : ]]) for s in self.states]
                self.next_states    = [torch.cat([ns[ : start_position], ns[end_position + 1 : ]]) for ns in self.next_states]
            else:
                self.states         = torch.cat([self.states[ : start_position], self.states[end_position + 1 : ]])
                self.next_states    = torch.cat([self.next_states[ : start_position], self.next_states[end_position + 1 : ]])

        elif start_position is not None and start_position > 0:            
            self.actions    = self.actions[ : start_position]
            self.rewards    = self.rewards[ : start_position]
            self.dones      = self.dones[ : start_position]            
            self.logprobs   = self.logprobs[ : start_position]

            if isinstance(self.states, list):
                self.states         = [s[ : start_position] for s in self.states]
                self.next_states    = [ns[ : start_position] for ns in self.next_states]
            else:
                self.states         = self.states[ : start_position]
                self.next_states    = self.next_states[ : start_position]

        elif end_position is not None and end_position != -1:
            self.actions        = self.actions[end_position + 1 : ]
            self.rewards        = self.rewards[end_position + 1 : ]
            self.dones          = self.dones[end_position + 1 : ]
            self.logprobs       = self.logprobs[end_position + 1 : ]

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

            del self.actions
            del self.rewards
            del self.dones
            del self.logprobs

            self.actions        = torch.tensor([])
            self.rewards        = torch.tensor([])
            self.dones          = torch.tensor([])
            self.logprobs       = torch.tensor([])            
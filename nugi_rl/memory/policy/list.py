import torch
from torch import device
from copy import deepcopy
from nugi_rl.memory.policy.base import Memory

class ListPolicyMemory(Memory):
    def __init__(self, device: device, capacity: int = 100000, datas: tuple = None):
        self.capacity       = capacity
        self.device         = device

        if datas is None:
            self.states         = []
            self.actions        = []
            self.rewards        = []
            self.dones          = []
            self.next_states    = []
            self.logprobs       = []
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states, self.logprobs = datas
            if len(self.dones) >= self.capacity:
                raise Exception('datas cannot be longer than capacity')

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return torch.tensor(self.states[idx], dtype = torch.float32, device = self.device), torch.tensor(self.actions[idx], dtype = torch.float32, device = self.device), \
            torch.tensor([self.rewards[idx]], dtype = torch.float32, device = self.device), torch.tensor([self.dones[idx]], dtype = torch.float32, device = self.device), \
            torch.tensor(self.next_states[idx], dtype = torch.float32, device = self.device), torch.tensor(self.logprobs[idx], dtype = torch.float32, device = self.device)

    def save(self, state: list, action: list, reward: float, done: bool, next_state: list, logprob: list) -> None:
        if len(self) >= self.capacity:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.dones[0]
            del self.next_states[0]
            del self.logprobs[0]

        self.states.append(deepcopy(state))
        self.actions.append(deepcopy(action))
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(deepcopy(next_state))
        self.logprobs.append(deepcopy(logprob))

    def get(self, start_position: int = 0, end_position: int = None) -> tuple:
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
        if end_position is not None and end_position != -1:
            del self.states[start_position : end_position + 1]
            del self.actions[start_position : end_position + 1]
            del self.rewards[start_position : end_position + 1]
            del self.dones[start_position : end_position + 1]
            del self.next_states[start_position : end_position + 1]
            del self.logprobs[start_position : end_position + 1]
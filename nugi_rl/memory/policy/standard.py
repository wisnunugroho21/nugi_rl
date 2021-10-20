from copy import deepcopy
import torch
from nugi_rl.memory.policy.base import Memory

class PolicyMemory(Memory):
    def __init__(self, capacity: int = 100000, datas: tuple = None):
        self.capacity       = capacity
        self.position       = 0

        if datas is None:
            self.states         = []
            self.actions        = []
            self.rewards        = []
            self.dones          = []
            self.next_states    = []
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states = datas
            if len(self.dones) >= self.capacity:
                raise Exception('datas cannot be longer than capacity')        

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return torch.tensor(self.states[idx], dtype = torch.float32), torch.tensor(self.actions[idx], dtype = torch.float32), \
            torch.tensor([self.rewards[idx]], dtype = torch.float32), torch.tensor([self.dones[idx]], dtype = torch.float32), \
            torch.tensor(self.next_states[idx], dtype = torch.float32)

    def save(self, state: list, action: list, reward: float, done: bool, next_state: list) -> None:
        if len(self) >= self.capacity:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.dones[0]
            del self.next_states[0]

        self.states.append(deepcopy(state))
        self.actions.append(deepcopy(action))
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(deepcopy(next_state))

    def get(self, start_position: int = 0, end_position: int = None) -> tuple:
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

    def clear(self, start_position: int = 0, end_position: int = None):
        if end_position is not None and end_position != -1:
            del self.states[start_position : end_position + 1]
            del self.actions[start_position : end_position + 1]
            del self.rewards[start_position : end_position + 1]
            del self.dones[start_position : end_position + 1]
            del self.next_states[start_position : end_position + 1]
        else:
            del self.states[start_position :]
            del self.actions[start_position :]
            del self.rewards[start_position :]
            del self.dones[start_position :]
            del self.next_states[start_position :]
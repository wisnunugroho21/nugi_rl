import torch
from memory.policy.standard import PolicyMemory

class TensorPolicyMemory(PolicyMemory):
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

    def save_obs(self, state, action, reward, done, next_states):
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
            self.next_states    = next_states.unsqueeze(0)
        else:
            self.states = torch.cat((self.states, state.unsqueeze(0)), dim = 0)
            self.actions = torch.cat((self.actions, action.unsqueeze(0)), dim = 0)
            self.rewards = torch.cat((self.rewards, reward.unsqueeze(0)), dim = 0)
            self.dones = torch.cat((self.dones, done.unsqueeze(0)), dim = 0)
            self.next_states = torch.cat((self.next_states, next_states.unsqueeze(0)), dim = 0)
        
    def clear_memory(self):
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

    def clear_idx(self, idx):
        self.states = torch.cat([self.states[0 : idx], self.states[idx+1 : ]])
        self.actions = torch.cat([self.actions[0 : idx], self.actions[idx+1 : ]])
        self.rewards = torch.cat([self.rewards[0 : idx], self.rewards[idx+1 : ]])
        self.dones = torch.cat([self.dones[0 : idx], self.dones[idx+1 : ]])
        self.next_states = torch.cat([self.next_states[0 : idx], self.next_states[idx+1 : ]])
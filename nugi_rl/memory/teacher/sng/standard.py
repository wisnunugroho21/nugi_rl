import torch
from torch.utils.data import Dataset

from nugi_rl.memory.teacher.sng.sng_template import SNGTemplateMemory

class SNMemory(Dataset):
    def __init__(self, capacity = 100000):
        self.capacity       = capacity

        self.policy_memory  = SNGTemplateMemory(capacity)
        self.expert_memory  = SNGTemplateMemory(capacity)

    def __len__(self):
        return len(self.expert_memory) if len(self.expert_memory) <= len(self.policy_memory) else len(self.policy_memory)

    def __getitem__(self, idx):
        policy_states, goals, policy_next_states = self.policy_memory[idx]
        expert_states, goals, expert_next_states = self.expert_memory[idx]

        return expert_states, expert_next_states, policy_states, policy_next_states, goals

    def save_policy_obs(self, state, goal, next_state):
        self.policy_memory.save_obs(state, goal, next_state)            

    def save_policy_replace_all(self, states, goals, next_states):
        self.policy_memory.save_replace_all(states, goals, next_states)

    def save_policy_all(self, states, goals, next_states):
        self.policy_memory.save_all(states, goals, next_states)

    def get_all_policy_items(self):         
        return self.policy_memory.get_all_items()

    def get_ranged_policy_items(self, start_position = 0, end_position = None):
        return self.policy_memory.get_ranged_items(start_position, end_position)

    def clear_policy_memory(self):
        self.policy_memory.clear_memory()

    def clear_policy_idx(self, idx):
        self.policy_memory.clear_idx(idx)


    def save_expert_obs(self, state, goal, next_state):
        self.expert_memory.save_obs(state, goal, next_state)

    def save_expert_replace_all(self, states, goals, next_states):
        self.expert_memory.save_replace_all(states, goals, next_states)

    def save_expert_all(self, states, goals, next_states):
        self.expert_memory.save_all(states, goals, next_states)

    def get_all_expert_items(self):         
        return self.expert_memory.get_all_items()

    def get_ranged_expert_items(self, start_position = 0, end_position = None):   
        return self.expert_memory.get_ranged_items(start_position, end_position)

    def clear_expert_memory(self):
        self.expert_memory.clear_memory()

    def clear_expert_idx(self, idx):
        self.expert_memory.clear_idx(idx)
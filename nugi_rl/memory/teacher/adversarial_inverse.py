import torch
from torch.utils.data import Dataset

from nugi_rl.memory.teacher.adversarial_inverse_template import AdvInvTemplateMemory

class AdvInvMemory(Dataset):
    def __init__(self, capacity = 100000):
        self.capacity       = capacity

        self.policy_memory  = AdvInvTemplateMemory(capacity)
        self.expert_memory  = AdvInvTemplateMemory(capacity)

    def __len__(self):
        return len(self.expert_memory) if len(self.expert_memory) <= len(self.policy_memory) else len(self.policy_memory)

    def __getitem__(self, idx):
        policy_states, policy_actions, policy_logprobs, policy_dones, policy_next_states = self.policy_memory[idx]
        expert_states, expert_actions, expert_logprobs, expert_dones, expert_next_states = self.expert_memory[idx]

        return policy_states, policy_actions, policy_logprobs, policy_dones, policy_next_states, \
            expert_states, expert_actions, expert_logprobs, expert_dones, expert_next_states

    def save_policy_obs(self, state, action, logprob, done, next_state):
        self.policy_memory.save_obs(state, action, logprob, done, next_state)            

    def save_policy_replace_all(self, states, actions, logprobs, dones, next_states):
        self.policy_memory.save_replace_all(states, actions, logprobs, dones, next_states)

    def save_policy_all(self, states, actions, logprobs, dones, next_states):
        self.policy_memory.save_all(states, actions, logprobs, dones, next_states)

    def get_all_policy_items(self):         
        return self.policy_memory.get_all_items()

    def get_ranged_policy_items(self, start_position = 0, end_position = None):
        return self.policy_memory.get_ranged_items(start_position, end_position)

    def clear_policy_memory(self):
        self.policy_memory.clear_memory()

    def clear_policy_idx(self, idx):
        self.policy_memory.clear_idx(idx)


    def save_expert_obs(self, state, action, logprob, done, next_state):
        self.expert_memory.save_obs(state, action, logprob, done, next_state)

    def save_expert_replace_all(self, states, actions, logprobs, dones, next_states):
        self.expert_memory.save_replace_all(states, actions, logprobs, dones, next_states)

    def save_expert_all(self, states, actions, logprobs, dones, next_states):
        self.expert_memory.save_all(states, actions, logprobs, dones, next_states)

    def get_all_expert_items(self):         
        return self.expert_memory.get_all_items()

    def get_ranged_expert_items(self, start_position = 0, end_position = None):   
        return self.expert_memory.get_ranged_items(start_position, end_position)

    def clear_expert_memory(self):
        self.expert_memory.clear_memory()

    def clear_expert_idx(self, idx):
        self.expert_memory.clear_idx(idx)
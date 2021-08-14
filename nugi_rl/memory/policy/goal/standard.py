import numpy as np
import torchvision.transforms as transforms
from memory.policy.standard import PolicyMemory

class GoalPolicyMemory(PolicyMemory):
    def __init__(self, datas = None):
        if datas is None :
            self.goals         = []
            super().__init__()

        else:
            states, goals, actions, rewards, dones, next_states = datas
            self.goals                 = goals

            super().__init__((states, actions, rewards, dones, next_states))

    def __getitem__(self, idx):
        states, actions, rewards, dones, next_states = super().__getitem__(idx)
        return states, self.goals[idx], actions, rewards, dones, next_states

    def save_obs(self, state, goal, action, reward, done, next_state):
        if len(self) >= self.capacity:
            del self.goals[0]

        super().save_obs(state, action, reward, done, next_state)
        self.goals.append(goal)

    def save_replace_all(self, states, goals, actions, rewards, dones, next_states):
        self.clear_memory()
        self.save_all(states, goals, actions, rewards, dones, next_states)

    def save_all(self, states, goals, actions, rewards, dones, next_states):
        for state, goal, action, reward, done, next_state in zip(states, goals, actions, rewards, dones, next_states):            
            self.save_obs(state, goal, action, reward, done, next_state)

    def get_all_items(self):
        states, actions, rewards, dones, next_states = super().get_all_items()
        return states, self.goals, actions, rewards, dones, next_states

    def get_ranged_items(self, start_position = 0, end_position = None):   
        if end_position is not None or end_position == -1:
            goals      = self.goals[start_position:end_position + 1]
        else:
            goals      = self.goals[start_position:]

        states, actions, rewards, dones, next_states = super().get_all_items()
        return states, goals, actions, rewards, dones, next_states

    def clear_memory(self):
        super().clear_memory()
        del self.goals[:]

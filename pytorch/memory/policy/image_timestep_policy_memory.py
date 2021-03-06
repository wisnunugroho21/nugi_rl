import numpy as np
from memory.policy.policy_memory import PolicyMemory

class ImageTimestepPolicyMemory(PolicyMemory):
    def __getitem__(self, idx):
        states      = self.states[idx].transpose(2, 3).transpose(1, 2)
        next_states = self.next_states[idx].transpose(2, 3).transpose(1, 2)

        return np.array(states, dtype = np.float32), np.array(self.actions[idx], dtype = np.float32), \
            np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), \
            np.array(next_states, dtype = np.float32)
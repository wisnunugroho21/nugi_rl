import numpy as np
from torch.utils.data import Dataset

class ImageStateClrMemory():
    def __init__(self, capacity = 10000):        
        self.capacity   = capacity
        self.states     = []
        self.images     = []
        self.position   = 0

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (np.array(self.images[idx], dtype = np.float32), np.array(self.states[idx], dtype = np.float32)) 

    def save_eps(self, data_state):
        image, state    = data_state

        if len(self.states) < self.capacity:
            self.states.append(None)
            self.images.append(None)

        self.states[self.position]  = state
        self.images[self.position]  = image

        self.position               = (self.position + 1) % self.capacity

    def save_replace_all(self, data_states):
        self.clear_memory()
        self.save_all(data_states)

    def save_all(self, data_states):
        images, states  = data_states
        for image, state in zip(images, states):
            if len(self.states) < self.capacity:
                self.states.append(None)
                self.images.append(None)

            self.states[self.position]  = state
            self.images[self.position]  = image

            self.position               = (self.position + 1) % self.capacity

    def get_all_items(self):
        return (self.images, self.states)

    def clear_memory(self):
        del self.images[:]
        del self.states[:]

import numpy as np
from torch.utils.data import Dataset

from memory.other.aux_memory import AuxMemory

class ImageStateAuxMemory(AuxMemory):
    def __init__(self):
        self.images = []
        super().__init__()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        states = super().__getitem__(idx)
        return (np.array(self.images[idx], dtype = np.float32), states)

    def save_eps(self, data_state):
        image, state    = data_state

        super().save_eps(state)
        self.images.append(image)

    def save_replace_all(self, data_states):
        images, states    = data_states

        super().save_all(states)
        self.images = images

    def save_all(self, data_states):
        images, states    = data_states
        
        super().save_all(states)
        self.images += images

    def get_all_items(self):
        states = super().get_all_items()
        return (self.images, states)

    def clear_memory(self):
        super().clear_memory()
        del self.images[:]

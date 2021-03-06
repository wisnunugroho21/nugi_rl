import numpy as np
from memory.aux_ppg.aux_memory import AuxMemory

class ImageTimestepAuxMemory(AuxMemory):
    def __getitem__(self, idx):
        states  = self.states[idx].transpose(2, 3).transpose(1, 2)
        return np.array(states, dtype = np.float32)
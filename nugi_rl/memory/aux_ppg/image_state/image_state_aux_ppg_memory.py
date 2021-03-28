import numpy as np
from memory.aux_ppg.aux_ppg_memory import AuxPpgMemory

class ImageStateAuxMemory(AuxPpgMemory):
    def __init__(self, datas = None):
        if datas is None :
            self.images         = []
            super().__init__()

        else:
            images, states              = datas
            self.images                 = images
            
            super().__init__((states))

    def __getitem__(self, idx):
        states = super().__getitem__(idx)
        return np.array(self.images[idx], dtype = np.float32), states

    def save_eps(self, image, state):
        if len(self) >= self.capacity:
            del self.images[0]

        super().save_eps(state)
        self.images.append(image)

    def save_replace_all(self, images, states):
        self.clear_memory()

        for image, state in zip(images, states):
            self.save_eps(image, state)

    def save_all(self, images, states):
        for image, state in zip(images, states):
            self.save_eps(image, state)

    def get_all_items(self):
        states = super().get_all_items()
        return self.images, states

    def clear_memory(self):
        super().clear_memory()
        del self.images[:]

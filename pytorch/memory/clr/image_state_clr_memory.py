import numpy as np
import torchvision.transforms as transforms
import torch

from memory.clr.clr_memory import ClrMemory

class ImageStateClrMemory(ClrMemory):
    def save_eps(self, data_state):
        image, _    = data_state
        super().save_eps(image)


    def save_all(self, data_states):
        images, _    = data_states

        for image in images:
            super().save_eps(image)

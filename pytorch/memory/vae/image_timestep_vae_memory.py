import torch
import numpy as np
from memory.vae.vae_memory import VaeMemory

class ImageTimestepVaeMemory(VaeMemory):
    def save_eps(self, images):
        for image in images:
            if len(self.images) < self.capacity:
                self.images.append(None)

            self.images[self.position]  = image
            self.position               = (self.position + 1) % self.capacity
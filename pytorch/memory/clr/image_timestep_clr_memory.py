import torch
import numpy as np
from memory.clr.clr_memory import ClrMemory

class ImageTimestepClrMemory(ClrMemory):
    def __getitem__(self, idx):
        images          = torch.FloatTensor(self.images[idx])

        crop_inputs     = self.trans_crop(images)
        jitter_inputs   = self.trans_jitter(images)

        return (np.array(crop_inputs.unsqueeze(0), dtype = np.float32), np.array(jitter_inputs.unsqueeze(0), dtype = np.float32))

    def save_eps(self, images):
        for image in images:
            if len(self.images) < self.capacity:
                self.images.append(None)

            self.images[self.position]  = image
            self.position               = (self.position + 1) % self.capacity
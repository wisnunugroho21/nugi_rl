import torch
import numpy as np
from memory.clr.clr_memory import ClrMemory

class ImageTimestepClrMemory(ClrMemory):
    def __getitem__(self, idx):
        images          = torch.FloatTensor(self.images[idx])

        crop_inputs     = self.trans_crop(images)
        jitter_inputs   = self.trans_jitter(images)

        return (crop_inputs.unsqueeze(0).detach().cpu().numpy(), jitter_inputs.unsqueeze(0).detach().cpu().numpy())

    def save_eps(self, images):
        for image in images:
            if len(self) >= self.capacity:
                del self.images[0]

            self.images.append(image)
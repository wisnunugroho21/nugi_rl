import torch
import numpy as np
from memory.clr.clr_memory import ClrMemory

class ImageTimestepClrMemory(ClrMemory):
    def __getitem__(self, idx):
        images          = torch.FloatTensor(self.images[idx])

        first_inputs    = self.first_trans(images)
        second_inputs   = self.second_trans(images)

        return (first_inputs.unsqueeze(0).detach().cpu().numpy(), second_inputs.unsqueeze(0).detach().cpu().numpy())

    def save_eps(self, images):
        for image in images:
            if len(self) >= self.capacity:
                del self.images[0]

            self.images.append(image)
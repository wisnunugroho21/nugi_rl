import torch
import numpy as np
from memory.clr.clr_memory import ClrMemory

class ImageTimestepClrMemory(ClrMemory):
    def __getitem__(self, idx):
        images          = torch.FloatTensor(self.images[idx])

        crop_inputs     = self.trans_crop(images)
        jitter_inputs   = self.trans_jitter(images)

        return (np.array(crop_inputs.unsqueeze(1), dtype = np.float32), np.array(jitter_inputs.unsqueeze(1), dtype = np.float32))
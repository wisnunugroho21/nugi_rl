import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ClrMemory(Dataset):
    def __init__(self, capacity = 10000, trans_crop = None, trans_jitter = None):        
        self.images         = []
        self.position       = 0
        self.capacity       = capacity
        self.trans_crop     = trans_crop
        self.trans_jitter   = trans_jitter

        if self.trans_crop is None:
            self.trans_crop = transforms.Compose([
                transforms.RandomCrop(128),
                transforms.Resize(160)
            ])

        if self.trans_jitter is None:
            self.trans_jitter = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2)
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images          = torch.FloatTensor(self.images[idx])

        crop_inputs     = self.trans_crop(images)
        jitter_inputs   = self.trans_jitter(images)

        return (np.array(crop_inputs, dtype = np.float32), np.array(jitter_inputs, dtype = np.float32))

    def save_eps(self, image):
        if len(self.images) < self.capacity:
            self.images.append(None)

        self.images[self.position]  = image
        self.position               = (self.position + 1) % self.capacity

    def save_replace_all(self, images):
        self.clear_memory()

        for image in images:
            self.save_eps(image)

    def save_all(self, images):
        for image in images:
            self.save_eps(image)

    def get_all_items(self):         
        return self.images

    def clear_memory(self):
        del self.images[:]
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VaeMemory(Dataset):
    def __init__(self, capacity = 10000):        
        self.images         = []
        self.position       = 0
        self.capacity       = capacity

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return np.array(self.images[idx], dtype = np.float32)

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
        self.position = 0
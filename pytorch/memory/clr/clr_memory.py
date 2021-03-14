import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ClrMemory(Dataset):
    def __init__(self, capacity = 10000, first_trans = None, second_trans = None):        
        self.images         = []
        self.capacity       = capacity
        self.first_trans    = first_trans
        self.second_trans   = second_trans

        if self.first_trans is None:
            self.first_trans = transforms.Compose([
                transforms.RandomCrop(270),
                transforms.Resize(320)
            ])

        if self.second_trans is None:
            self.second_trans = transforms.Compose([                
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p = 0.8),
                transforms.RandomGrayscale(p = 0.2)
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images          = torch.FloatTensor(self.images[idx])

        first_inputs    = self.first_trans(images)
        second_inputs   = self.second_trans(images)

        return (first_inputs.detach().cpu().numpy(), second_inputs.detach().cpu().numpy())

    def save_eps(self, state):
        if len(self) >= self.capacity:
            del self.images[0]

        self.images.append(state)

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
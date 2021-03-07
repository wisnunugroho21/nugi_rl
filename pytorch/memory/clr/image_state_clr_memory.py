import numpy as np
import torchvision.transforms as transforms
import torch

class ImageStateClrMemory():
    def __init__(self, capacity = 10000, trans_crop = None, trans_jitter = None):        
        self.images         = []
        self.states         = []
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
        return len(self.states)

    def __getitem__(self, idx):
        images          = torch.FloatTensor(self.images[idx])

        crop_inputs     = self.trans_crop(images)
        jitter_inputs   = self.trans_jitter(images)

        return ((np.array(crop_inputs, dtype = np.float32), self.states[idx]), (np.array(jitter_inputs, dtype = np.float32), self.states[idx]))

    def save_eps(self, data_state):
        image, state    = data_state        

        if len(self.states) < self.capacity:
            self.states.append(None)
            self.images.append(None)

        self.states[self.position]  = state
        self.images[self.position]  = image

        self.position   = (self.position + 1) % self.capacity

    def save_replace_all(self, data_states):
        self.clear_memory()
        images, states  = data_states

        for image, state in zip(images, states):
            if len(self.states) < self.capacity:
                self.states.append(None)
                self.images.append(None)

            self.states[self.position]  = state
            self.images[self.position]  = image
        
            self.position   = (self.position + 1) % self.capacity        

    def save_all(self, data_states):
        images, states  = data_states

        for image, state in zip(images, states):
            if len(self.states) < self.capacity:
                self.states.append(None)
                self.images.append(None)

            self.states[self.position]  = state
            self.images[self.position]  = image
        
            self.position   = (self.position + 1) % self.capacity

    def get_all_items(self):
        images = super().get_all_items()
        return (images, self.states)

    def clear_memory(self):
        super().clear_memory()
        del self.states[:]

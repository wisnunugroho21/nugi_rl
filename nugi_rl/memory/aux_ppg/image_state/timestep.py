import torch
import torchvision.transforms as transforms
from memory.aux_ppg.standard import AuxPpgMemory

class TimeImageStateAuxPpgMemory(AuxPpgMemory):
    def __init__(self, datas = None):
        if datas is None :
            self.images         = []
            super().__init__()

        else:
            images, states              = datas
            self.images                 = images
            
            super().__init__((states))

        self.trans  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        images  = torch.stack([self.trans(image) for image in self.images[idx]])

        states = super().__getitem__(idx)
        return images, states

    def save_obs(self, image, state):
        if len(self) >= self.capacity:
            del self.images[0]

        super().save_obs(state)
        self.images.append(image)

    def save_replace_all(self, images, states):
        self.clear_memory()

        for image, state in zip(images, states):
            self.save_obs(image, state)

    def save_all(self, images, states):
        for image, state in zip(images, states):
            self.save_obs(image, state)

    def get_all_items(self):
        states = super().get_all_items()
        return self.images, states

    def get_ranged_items(self, start_position = 0, end_position = None):   
        if end_position is not None or -1:
            images  = self.images[start_position:end_position + 1]
        else:
            images  = self.images[start_position:]

        states = super().get_all_items()
        return images, states

    def clear_memory(self):
        super().clear_memory()
        del self.images[:]

    def transform(self, images):
        return torch.stack([self.trans(image) for image in images])

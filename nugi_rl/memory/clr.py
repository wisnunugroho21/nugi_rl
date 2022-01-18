from copy import deepcopy
from torchvision.transforms import Compose
import torchvision.transforms as transforms

from memory.base import Memory

class ClrMemory(Memory):
    def __init__(self, input_trans: Compose = None, target_trans: Compose = None, capacity: int = 10000):        
        self.images         = []

        self.capacity       = capacity
        self.input_trans    = input_trans
        self.target_trans   = target_trans

        if self.input_trans is None:
            self.input_trans = transforms.Compose([
                transforms.RandomResizedCrop(320),                           
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p = 0.8),
                transforms.RandomGrayscale(p = 0.2),
                transforms.GaussianBlur(33),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if self.target_trans is None:
            self.target_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images          = self.images[idx]

        input_images    = self.input_trans(images)
        target_images   = self.target_trans(images)

        return input_images, target_images

    def save(self, image) -> None:
        if len(self) >= self.capacity:
            del self.images[0]

        self.images.append(deepcopy(image))

    def save_all(self, images) -> None:
        for image in images:
            self.save(image)

    def get(self, start_position: int = 0, end_position: int = None) -> tuple:
        if end_position is not None or end_position == -1:
            images  = self.images[start_position:end_position + 1]
        else:
            images  = self.images[start_position:]

        return images

    def clear(self, start_position: int = 0, end_position: int = None):
        if end_position is not None and end_position != -1:
            del self.images[start_position : end_position + 1]
        else:
            del self.images[start_position :]
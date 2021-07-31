from copy import deepcopy
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class AuxClrMemory(Dataset):
    def __init__(self, capacity = 10000, input_trans = None, target_trans = None):        
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

    def save_obs(self, image):
        if len(self) >= self.capacity:
            del self.images[0]

        self.images.append(deepcopy(image))

    def save_replace_all(self, images):
        self.clear_memory()

        for image in images:
            self.save_obs(image)

    def save_all(self, images):
        for image in images:
            self.save_obs(image)

    def get_all_items(self):         
        return self.images

    def get_ranged_items(self, start_position = 0, end_position = None):   
        if end_position is not None or -1:
            images  = self.images[start_position:end_position + 1]
        else:
            images  = self.images[start_position:]

        return images

    def clear_memory(self):
        del self.images[:]

    def clear_idx(self, idx):
        del self.images[idx]
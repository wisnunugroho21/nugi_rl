from torch.utils.data import Dataset

class ClrMemory(Dataset):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def save(self, image) -> None:
        raise NotImplementedError

    def save_all(self, images):
        for image in images:
            self.save(image)

    def get(self, start_position: int = 0, end_position: int = None):
        raise NotImplementedError

    def clear(self, start_position: int = 0, end_position: int = None) -> None:
        raise NotImplementedError
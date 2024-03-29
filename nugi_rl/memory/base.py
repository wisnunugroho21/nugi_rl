from torch.utils.data import Dataset

class Memory(Dataset):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    def save_all(self) -> None:
        raise NotImplementedError

    def get(self, start_position: int = 0, end_position: int = None):
        raise NotImplementedError

    def clear(self, start_position: int = 0, end_position: int = None) -> None:
        raise NotImplementedError
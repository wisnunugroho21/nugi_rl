from torch.utils.data import Dataset

class Memory(Dataset):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def save(self, state: list, action: list, reward: float, done: bool, next_state: list):
        raise NotImplementedError

    def get(self, start_position: int = 0, end_position: int = None) -> tuple:
        raise NotImplementedError

    def clear(self, start_position: int = 0, end_position: int = None):
        raise NotImplementedError
from torch.utils.data import Dataset


class Memory(Dataset):
    def __len__(self) -> int:
        raise NotImplementedError

    def clear(self, start_position: int = 0, end_position: int | None = None) -> None:
        raise NotImplementedError

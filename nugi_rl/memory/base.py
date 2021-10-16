from torch.utils.data import Dataset

class Memory(Dataset):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def save_obs(self, state, action, reward, done, next_state):
        raise NotImplementedError

    def clear_memory(self):
        raise NotImplementedError
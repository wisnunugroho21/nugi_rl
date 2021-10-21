from torch.utils.data import Dataset

class Memory(Dataset):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def save(self, state, action, reward, done, next_state) -> None:
        raise NotImplementedError

    def save_all(self, states, actions, rewards, dones, next_states):
        for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
            self.save(state, action, reward, done, next_state)

    def get(self, start_position: int = 0, end_position: int = None):
        raise NotImplementedError

    def clear(self, start_position: int = 0, end_position: int = None) -> None:
        raise NotImplementedError
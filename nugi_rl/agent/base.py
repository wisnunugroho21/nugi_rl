class Agent():
    def act(self, state: list) -> list:
        raise NotImplementedError

    def save_obs(self, state: list, action: list, reward: float, done: bool, next_state: list):
        raise NotImplementedError
        
    def update(self):
        raise NotImplementedError

    def get_obs(self, start_idx: int = None, end_idx: int = None) -> tuple:
        raise NotImplementedError

    def load_weights(self):
        raise NotImplementedError

    def save_weights(self):
        raise NotImplementedError
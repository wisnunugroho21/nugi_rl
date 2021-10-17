class Agent():
    def act(self, state: list) -> list:
        raise NotImplementedError

    def logprob(self, state: list, action: list) -> list:
        raise NotImplementedError

    def save_obs(self, state: list, action: list, reward: float, done: bool, next_state: list):
        raise NotImplementedError
        
    def update(self) -> None:
        raise NotImplementedError

    def get_obs(self, start_position: int = None, end_position: int = None) -> tuple:
        raise NotImplementedError

    def load_weights(self) -> None:
        raise NotImplementedError

    def save_weights(self) -> None:
        raise NotImplementedError
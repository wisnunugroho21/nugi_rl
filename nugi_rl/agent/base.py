class Agent():
    def act(self, state):
        raise NotImplementedError

    def logprob(self, state, action):
        raise NotImplementedError

    def save_obs(self, state, action, reward, done, next_state):
        raise NotImplementedError
        
    def update(self) -> None:
        raise NotImplementedError

    def get_obs(self, start_position: int = None, end_position: int = None):
        raise NotImplementedError

    def load_weights(self) -> None:
        raise NotImplementedError

    def save_weights(self) -> None:
        raise NotImplementedError
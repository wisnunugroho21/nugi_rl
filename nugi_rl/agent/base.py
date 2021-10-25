class Agent():
    def act(self, state):
        raise NotImplementedError

    def logprob(self, state, action):
        raise NotImplementedError

    def save_obs(self, state, action, reward, done, next_state)-> None:
        raise NotImplementedError

    def get_obs(self, start_position: int = 0, end_position: int = None):
        raise NotImplementedError

    def clear_obs(self, start_position: int = 0, end_position: int = None) -> None:
        raise NotImplementedError
        
    def update(self) -> None:
        raise NotImplementedError    

    def load_weights(self) -> None:
        raise NotImplementedError

    def save_weights(self) -> None:
        raise NotImplementedError
class Agent():
    def act(self, state):
        raise NotImplementedError

    def save_obs(self, state, action, reward, done, next_state):
        raise NotImplementedError
        
    def update(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError
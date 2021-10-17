class Environment():
    def is_discrete(self):
        raise NotImplementedError

    def get_obs_dim(self):
        raise NotImplementedError
            
    def get_action_dim(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self) -> None:
        raise NotImplementedError
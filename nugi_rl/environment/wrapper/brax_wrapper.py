import jax
import numpy as onp

class BraxWrapper():
    def __init__(self, env):
        self.env = env        

    def is_discrete(self):
        return False

    def get_obs_dim(self):
        return self.env.observation_size
            
    def get_action_dim(self):
        return self.env.action_size

    def reset(self):
        tracjectory = self.env.reset(rng=jax.random.PRNGKey(seed=0))
        return onp.array(tracjectory.obs).tolist()

    def step(self, action):
        tracjectory = self.env.step(action)
        return onp.array(tracjectory.obs).tolist(), onp.array(tracjectory.reward).tolist(), onp.array(tracjectory.done).tolist(), []

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
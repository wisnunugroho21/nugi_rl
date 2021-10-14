import gym
import torch

from brax.envs.to_torch import JaxToTorchWrapper
from functools import partial
from brax.envs import _envs, create_gym_env

class BraxWrapper():
    def __init__(self, env):
        self.env = JaxToTorchWrapper(env)

    def register_brax_gym(gym):
        CUDA_AVAILABLE = torch.cuda.is_available()
        if CUDA_AVAILABLE:
            # BUG: (@lebrice): Getting a weird "CUDA error: out of memory" RuntimeError
            # during JIT, which can be "fixed" by first creating a dummy cuda tensor!
            v = torch.ones(1, device="cuda")

        for env_name, env_class in _envs.items():
            env_id = f"brax_{env_name}-v0"
            entry_point = partial(create_gym_env, env_name=env_name)
            if env_id not in gym.envs.registry.env_specs:
                print(f"Registring brax's '{env_name}' env under id '{env_id}'.")
                gym.register(env_id, entry_point=entry_point)        

    def is_discrete(self):
        return type(self.env.action_space) is not gym.spaces.Box

    def get_obs_dim(self):
        if type(self.env.observation_space) is gym.spaces.Box:
            state_dim = 1

            if len(self.env.observation_space.shape) > 1:                
                for i in range(len(self.env.observation_space.shape)):
                    state_dim *= self.env.observation_space.shape[i]            
            else:
                state_dim = self.env.observation_space.shape[0]

            return state_dim
        else:
            return self.env.observation_space.n
            
    def get_action_dim(self):
        if self.is_discrete():
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
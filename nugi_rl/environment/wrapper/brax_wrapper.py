import gym
import torch

from brax.envs.to_torch import JaxToTorchWrapper
from functools import partial
from brax.envs import _envs, create_gym_env

class BraxWrapper():
    def __init__(self, env):
        self.env = JaxToTorchWrapper(env)

    def register_brax_gym(gym):
        for env_name, env_class in _envs.items():
            env_id = f"brax_{env_name}-v0"
            entry_point = partial(create_gym_env, env_name=env_name)
            if env_id not in gym.envs.registry.env_specs:
                print(f"Registring brax's '{env_name}' env under id '{env_id}'.")
                gym.register(env_id, entry_point=entry_point)        

    def is_discrete(self):
        return type(self.env.action_space) is not gym.spaces.Box

    def get_obs_dim(self):
        if self.is_discrete():
            return self.env.observation_space.n
        else:
            return self.env.observation_space.shape[-1]
            
    def get_action_dim(self):
        if self.is_discrete():
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[-1]

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
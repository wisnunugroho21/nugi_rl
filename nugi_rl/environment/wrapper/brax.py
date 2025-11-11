from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper

from nugi_rl.environment.wrapper.gym import GymWrapper


class BraxWrapper(GymWrapper):
    def __init__(self, env_name: str, episode_length: int):
        env = envs.create(env_name, episode_length=episode_length)
        env = gym_wrapper.GymWrapper(env)
        self.env = torch_wrapper.TorchWrapper(env)

import gym

from brax.envs.to_torch import JaxToTorchWrapper
from brax.envs import _envs, create_gym_env
from functools import partial

from nugi_rl.environment.wrapper.gym import GymWrapper

class BraxWrapper(GymWrapper):
    def __init__(self, env_name, episode_length):
        self._register_brax_gym()

        env = gym.make(env_name, episode_length = episode_length)
        self.env = JaxToTorchWrapper(env)        

    def _register_brax_gym(self):
        for env_name, env_class in _envs.items():
            env_id      = f"brax_{env_name}-v0"
            entry_point = partial(create_gym_env, env_name=env_name)

            if env_id not in gym.envs.registry.env_specs:
                print(f"Registring brax's '{env_name}' env under id '{env_id}'.")
                gym.register(env_id, entry_point=entry_point)
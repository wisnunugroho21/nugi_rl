from torch import Tensor
from nugi_rl.memory.base import Memory
from typing import List, Union

class Agent():
    def act(self, state: Tensor) -> Tensor:
        raise NotImplementedError

    def logprob(self, state: Tensor, action: Tensor) -> Tensor:
        raise NotImplementedError

    def save_obs(self, state: Union[Tensor, List[Tensor]], action: Tensor, reward: Tensor, done: Tensor, next_state: Union[Tensor, List[Tensor]], logprob: Tensor)-> None:
        raise NotImplementedError

    def save_all(self, states: Union[Tensor, List[Tensor]], actions: Tensor, rewards: Tensor, dones: Tensor, next_states: Union[Tensor, List[Tensor]], logprobs: Tensor) -> None:
        raise NotImplementedError

    def update(self) -> None:
        raise NotImplementedError

    def get_obs(self, start_position: int = 0, end_position: int = None):
        raise NotImplementedError

    def clear_obs(self, start_position: int = 0, end_position: int = None) -> None:
        raise NotImplementedError

    def load_weights(self) -> None:
        raise NotImplementedError

    def save_weights(self) -> None:
        raise NotImplementedError
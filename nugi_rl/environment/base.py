from torch import Tensor

class Environment():
    def is_discrete(self) -> bool:
        raise NotImplementedError

    def get_obs_dim(self) -> int:
        raise NotImplementedError
            
    def get_action_dim(self) -> int:
        raise NotImplementedError

    def reset(self) -> Tensor:
        raise NotImplementedError

    def step(self, action: Tensor) -> Tensor:
        raise NotImplementedError

    def render(self) -> None:
        raise NotImplementedError
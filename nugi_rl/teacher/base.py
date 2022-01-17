from torch import Tensor

class Teacher():
    def teach(self, state: Tensor, next_state: Tensor, goal: Tensor) -> Tensor:
        raise NotImplementedError

    def save_obs(self, state: Tensor, goal: Tensor, next_state: Tensor)-> None:
        raise NotImplementedError

    def save_all(self, states: Tensor, goals: Tensor, next_states: Tensor) -> None:
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
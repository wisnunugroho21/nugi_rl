from torch import Tensor


class Agent:
    def act(self, state: Tensor) -> Tensor:
        raise NotImplementedError

    def logprob(self, state: Tensor, action: Tensor) -> Tensor:
        raise NotImplementedError

    def update(self, config: str = "") -> None:
        raise NotImplementedError

    def save_obs(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        next_state: Tensor,
        logprob: Tensor,
    ) -> None:
        raise NotImplementedError

    def save_all(
        self,
        states: list[Tensor],
        actions: list[Tensor],
        rewards: list[Tensor],
        dones: list[Tensor],
        next_states: list[Tensor],
        logprobs: list[Tensor],
    ) -> None:
        raise NotImplementedError

    def get_obs(
        self, start_position: int = 0, end_position: int | None = None
    ) -> tuple[
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
    ]:
        raise NotImplementedError

    def clear_obs(
        self, start_position: int = 0, end_position: int | None = None
    ) -> None:
        raise NotImplementedError

    def load_weights(self) -> None:
        raise NotImplementedError

    def save_weights(self) -> None:
        raise NotImplementedError

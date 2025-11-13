from torch import Tensor

from nugi_rl.memory.base import Memory


class PolicyMemory(Memory):
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(
        self, idx: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def save(
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
        for state, action, reward, done, next_state, logprob in zip(
            states, actions, rewards, dones, next_states, logprobs
        ):
            self.save(state, action, reward, done, next_state, logprob)

    def get(
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

    def clear(self, start_position: int = 0, end_position: int | None = None) -> None:
        raise NotImplementedError

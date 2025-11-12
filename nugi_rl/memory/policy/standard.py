from torch import Tensor

from nugi_rl.memory.policy.base import PolicyMemory


class StandardPolicyMemory(PolicyMemory):
    def __init__(self, capacity=1000000):
        self.capacity = capacity

        self.states: list[Tensor] = []
        self.actions: list[Tensor] = []
        self.rewards: list[Tensor] = []
        self.dones: list[Tensor] = []
        self.next_states: list[Tensor] = []
        self.logprobs: list[Tensor] = []

    def __len__(self) -> int:
        return len(self.dones)

    def __getitem__(
        self, idx: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx].unsqueeze(-1),
            self.dones[idx].unsqueeze(-1),
            self.next_states[idx],
            self.logprobs[idx],
        )

    def save(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        next_state: Tensor,
        logprob: Tensor,
    ) -> None:
        if len(self) >= self.capacity:
            self.states = self.states[1:]
            self.actions = self.actions[1:]
            self.rewards = self.rewards[1:]
            self.dones = self.dones[1:]
            self.next_states = self.next_states[1:]
            self.logprobs = self.logprobs[1:]

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.logprobs.append(logprob)

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
        if end_position is not None and end_position != -1:
            states = self.states[start_position : end_position + 1]
            actions = self.actions[start_position : end_position + 1]
            rewards = self.rewards[start_position : end_position + 1]
            dones = self.dones[start_position : end_position + 1]
            next_states = self.next_states[start_position : end_position + 1]
            logprobs = self.logprobs[start_position : end_position + 1]

        else:
            states = self.states[start_position:]
            actions = self.actions[start_position:]
            rewards = self.rewards[start_position:]
            dones = self.dones[start_position:]
            next_states = self.next_states[start_position:]
            logprobs = self.logprobs[start_position:]

        return states, actions, rewards, dones, next_states, logprobs

    def clear(self, start_position: int = 0, end_position: int | None = None) -> None:
        if (
            start_position is not None
            and start_position > 0
            and end_position is not None
            and end_position != -1
        ):
            self.states = [
                *self.states[:start_position],
                *self.states[end_position + 1 :],
            ]

            self.actions = [
                *self.actions[:start_position],
                *self.actions[end_position + 1 :],
            ]

            self.rewards = [
                *self.rewards[:start_position],
                *self.rewards[end_position + 1 :],
            ]

            self.dones = [
                *self.dones[:start_position],
                *self.dones[end_position + 1 :],
            ]

            self.next_states = [
                *self.next_states[:start_position],
                *self.next_states[end_position + 1 :],
            ]

            self.logprobs = [
                *self.logprobs[:start_position],
                *self.logprobs[end_position + 1 :],
            ]

        elif start_position is not None and start_position > 0:
            self.states = self.states[:start_position]
            self.actions = self.actions[:start_position]
            self.rewards = self.rewards[:start_position]
            self.dones = self.dones[:start_position]
            self.next_states = self.next_states[:start_position]
            self.logprobs = self.logprobs[:start_position]

        elif end_position is not None and end_position != -1:
            self.states = self.states[end_position + 1 :]
            self.actions = self.actions[end_position + 1 :]
            self.rewards = self.rewards[end_position + 1 :]
            self.dones = self.dones[end_position + 1 :]
            self.next_states = self.next_states[end_position + 1 :]
            self.logprobs = self.logprobs[end_position + 1 :]

        else:
            del self.states
            del self.actions
            del self.rewards
            del self.dones
            del self.next_states
            del self.logprobs

            self.states = []
            self.actions = []
            self.rewards = []
            self.dones = []
            self.next_states = []
            self.logprobs = []

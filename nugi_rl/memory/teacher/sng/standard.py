from torch import Tensor

from nugi_rl.memory.base import Memory
from nugi_rl.memory.teacher.sng.template import SNGTemplateMemory


class SNGMemory(Memory):
    def __init__(self, capacity=1000000):
        self.capacity = capacity

        self.policy_memory = SNGTemplateMemory(capacity)
        self.expert_memory = SNGTemplateMemory(capacity)

    def __len__(self):
        return (
            len(self.expert_memory)
            if len(self.expert_memory) <= len(self.policy_memory)
            else len(self.policy_memory)
        )

    def __getitem__(self, idx):
        policy_states, goals, policy_next_states = self.policy_memory[idx]
        expert_states, goals, expert_next_states = self.expert_memory[idx]

        return (
            expert_states,
            expert_next_states,
            policy_states,
            policy_next_states,
            goals,
        )

    def save(
        self,
        state: Tensor,
        goal: Tensor,
        next_state: Tensor,
    ) -> None:
        self.save_policy(state, goal, next_state)
        self.save_expert(state, goal, next_state)

    def save_all(
        self,
        states: list[Tensor],
        goals: list[Tensor],
        next_states: list[Tensor],
    ) -> None:
        self.save_all_policy(states, goals, next_states)
        self.save_all_expert(states, goals, next_states)

    def save_policy(
        self,
        state: Tensor,
        goal: Tensor,
        next_state: Tensor,
    ) -> None:
        self.policy_memory.save(state, goal, next_state)

    def save_all_policy(
        self,
        states: list[Tensor],
        goals: list[Tensor],
        next_states: list[Tensor],
    ) -> None:
        self.policy_memory.save_all(states, goals, next_states)

    def get_policy(self, start_position: int = 0, end_position: int | None = None):
        return self.policy_memory.get(start_position, end_position)

    def clear_policy(
        self, start_position: int = 0, end_position: int | None = None
    ) -> None:
        self.policy_memory.clear(start_position, end_position)

    def save_expert(
        self,
        state: Tensor,
        goal: Tensor,
        next_state: Tensor,
    ) -> None:
        self.expert_memory.save(state, goal, next_state)

    def save_all_expert(
        self,
        states: list[Tensor],
        goals: list[Tensor],
        next_states: list[Tensor],
    ) -> None:
        self.expert_memory.save_all(states, goals, next_states)

    def get_expert(self, start_position: int = 0, end_position: int | None = None):
        return self.expert_memory.get(start_position, end_position)

    def clear_expert(
        self, start_position: int = 0, end_position: int | None = None
    ) -> None:
        self.expert_memory.clear(start_position, end_position)

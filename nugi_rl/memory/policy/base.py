from nugi_rl.memory.base import Memory

class PolicyMemory(Memory):
    def save(self, state, action, reward, done, next_state, logprob) -> None:
        raise NotImplementedError

    def save_all(self, states, actions, rewards, dones, next_states, logprobs) -> None:
        for state, action, reward, done, next_state, logprob in zip(states, actions, rewards, dones, next_states, logprobs):
            self.save(state, action, reward, done, next_state, logprob)

    def get(self, start_position: int = 0, end_position: int = None) -> tuple:
        raise NotImplementedError
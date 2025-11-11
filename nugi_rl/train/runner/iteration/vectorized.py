import torch

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.train.runner.iteration.standard import IterRunner
from nugi_rl.utilities.plotter.base import Plotter


class VectorizedRunner(IterRunner):
    def __init__(
        self,
        agent: Agent,
        envs: list[Environment],
        is_save_memory: bool,
        render: bool,
        n_update: int,
        plotter: Plotter | None = None,
        n_plot_batch: int = 1,
    ) -> None:
        self.agent = agent
        self.env = envs
        self.plotter = plotter

        self.render = render
        self.is_save_memory = is_save_memory
        self.n_update = n_update
        self.n_plot_batch = n_plot_batch

        self.t_updates = torch.zeros(len(self.env))
        self.i_episode = torch.zeros(len(self.env))
        self.total_reward = torch.zeros(len(self.env))
        self.eps_time = torch.zeros(len(self.env))

        self.states = torch.stack([env.reset() for env in self.env])

    def run(self) -> tuple:
        for _ in range(self.n_update):
            actions = self.agent.act(self.states)
            logprobs = self.agent.logprob(self.states, actions)

            for index, (env, action, logprob) in enumerate(
                zip(self.env, actions, logprobs)
            ):
                next_state, reward, done = env.step(action)

                if self.is_save_memory:
                    self.agent.save_obs(
                        self.states,
                        action,
                        reward,
                        done,
                        next_state,
                        logprob,
                    )

                self.states[index] = next_state
                self.eps_time[index] += 1
                self.total_reward[index] += reward

                if self.render:
                    env.render()

                if done:
                    self.i_episode += 1

                    print(
                        "Agent {} Episode {} \t t_reward: {} \t eps time: {}".format(
                            index,
                            self.i_episode[index],
                            self.total_reward[index],
                            self.eps_time[index],
                        )
                    )

                    if (
                        self.plotter is not None
                        and self.i_episode % self.n_plot_batch == 0
                    ):
                        self.plotter.plot(
                            {"Rewards": self.total_reward, "Times": self.eps_time}
                        )

                    self.states[index] = env.reset()
                    self.total_reward[index] = 0
                    self.eps_time[index] = 0

        return self.agent.get_obs(-1 * self.n_update * len(self.env))

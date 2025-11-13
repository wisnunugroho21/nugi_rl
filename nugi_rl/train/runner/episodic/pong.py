import torch
from torch import Tensor

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.helpers.math import prepro_half_one_dim
from nugi_rl.train.runner.episodic.standard import EpisodicRunner
from nugi_rl.utilities.plotter.base import Plotter


class PongRunner(EpisodicRunner):
    def __init__(
        self,
        agent: Agent,
        env: Environment,
        is_save_memory: bool,
        render: bool,
        n_update: int,
        plotter: Plotter | None = None,
        n_plot_batch: int = 100,
    ):
        super().__init__(
            agent, env, is_save_memory, render, n_update, plotter, n_plot_batch
        )

    def run(
        self,
    ) -> tuple[
        list[Tensor] | list[list[Tensor]],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor] | list[list[Tensor]],
        list[Tensor],
    ]:
        done = False
        total_reward = 0
        eps_time = 0

        for _ in range(self.n_update):
            initial_obs = self.env.reset()
            if isinstance(initial_obs, list):
                obs = initial_obs[0]
            else:
                obs = initial_obs

            obs = prepro_half_one_dim(obs)
            state = obs

            while not done:
                action = self.agent.act(state)
                logprob = self.agent.logprob(state, action)

                action_gym = torch.where(action != 0, action + 1, action)

                next_obs, reward, done = self.env.step(action_gym)
                if isinstance(next_obs, list):
                    next_obs_tensor = next_obs[0]
                else:
                    next_obs_tensor = next_obs

                next_obs_tensor = prepro_half_one_dim(next_obs_tensor)
                next_state = next_obs_tensor - obs

                if self.is_save_memory:
                    self.agent.save_obs(
                        state, action, reward, done, next_state, logprob
                    )

                state = next_state
                obs = next_obs_tensor
                eps_time += 1
                total_reward += reward

                if self.render:
                    self.env.render()

            self.i_episode += 1
            print(
                "Episode {} \t t_reward: {} \t time: {} ".format(
                    self.i_episode, total_reward, eps_time
                )
            )

            if self.plotter is not None and self.i_episode % self.n_plot_batch == 0:
                self.plotter.plot({"Rewards": total_reward, "Times": eps_time})

        return self.agent.get_obs(-eps_time)

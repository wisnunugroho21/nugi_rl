import torch

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.train.runner.iteration.standard import IterRunner

class VectorizedRunner(IterRunner):
    def __init__(self, agent: Agent, envs: list[Environment], is_save_memory: bool, render: bool, n_update: int, n_plot_batch: int = 100) -> None:
        super().__init__(agent, envs, is_save_memory, render, n_update, plotter, n_plot_batch)

        self.t_updates          = [0 for _ in self.env]
        self.i_episode          = [0 for _ in self.env]
        self.total_reward       = [0 for _ in self.env]
        self.eps_time           = [0 for _ in self.env]

        self.states             = [env.reset() for env in self.env]

    def run(self) -> tuple:
        for _ in range(self.n_update):
            actions                     = self.agent.act(torch.stack(self.states))
            logprobs                    = self.agent.logprob(torch.stack(self.states), actions)

            for index, (env, action, logprob) in enumerate(zip(self.env, actions, logprobs)):
                next_state, reward, done, _ = env.step(action)
                
                if self.is_save_memory:
                    self.agent.save_obs(torch.stack(self.states), action, reward, done, next_state, logprob)
                    
                self.states[index]          = next_state
                self.eps_time[index]        += 1 
                self.total_reward[index]    += reward
                        
                if self.render:
                    env.render()

                if done:                
                    self.i_episode  += 1

                    print('Agent {} Episode {} \t t_reward: {} \t eps time: {}'.format(index, self.i_episode[index], self.total_reward[index], self.eps_time[index]))

                    if self.plotter is not None and self.i_episode % self.n_plot_batch == 0:
                        self.plotter.plot({
                            'Rewards': self.total_reward,
                            'Times': self.eps_time
                        })

                    self.states         = env.reset()
                    self.total_reward   = 0
                    self.eps_time       = 0    

        return self.agent.get_obs(-1 * self.n_update * len(self.envs))
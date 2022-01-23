import torch

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.helpers.math import prepro_half_one_dim
from nugi_rl.utilities.plotter.base import Plotter
from nugi_rl.train.runner.iteration.standard import IterRunner

class PongRunner(IterRunner):
    def __init__(self, agent: Agent, env: Environment, is_save_memory: bool, render: bool, n_update: int, plotter: Plotter = None, n_plot_batch: int = 100):
        super().__init__(agent, env, is_save_memory, render, n_update, plotter, n_plot_batch)

        obs         = self.env.reset()  
        self.obs    = prepro_half_one_dim(obs)
        self.states = self.obs

    def run(self) -> tuple:
        for _ in range(self.n_update):
            action      = self.agent.act(self.states)
            logprob     = self.agent.logprob(self.states, action)

            action_gym  = action + 1 if action != 0 else torch.tensor(0)

            next_obs, reward, done, _ = self.env.step(action_gym)
            next_obs    = prepro_half_one_dim(next_obs)
            next_state  = next_obs - self.obs
            
            if self.is_save_memory:
                self.agent.save_obs(self.states, action, reward, done, next_state, logprob)
                
            self.states         = next_state
            self.obs            = next_obs
            self.eps_time       += 1 
            self.total_reward   += reward
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1

                print('Episode {} \t t_reward: {} \t eps time: {}'.format(self.i_episode, self.total_reward, self.eps_time))

                if self.i_episode % self.n_plot_batch == 0 and self.plotter is not None:
                    self.plotter.plot({
                        'Rewards': self.total_reward,
                        'Times': self.eps_time
                    })

                obs         = self.env.reset()  
                self.obs    = prepro_half_one_dim(obs)
                self.states = self.obs

                self.total_reward   = 0
                self.eps_time       = 0

        return self.agent.get_obs(-self.n_update)
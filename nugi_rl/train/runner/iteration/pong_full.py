import numpy as np
import torch

from nugi_rl.train.runner.iteration.standard import IterRunner

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.helpers.math import prepro_half_one_dim
from nugi_rl.utilities.plotter.base import Plotter
from nugi_rl.train.runner.iteration.standard import IterRunner

class PongFullRunner(IterRunner):
    def __init__(self, agent: Agent, env: Environment, is_save_memory: bool, render: bool, n_update: int, plotter: Plotter = None, n_plot_batch: int = 100):
        super().__init__(agent, env, is_save_memory, render, n_update, plotter, n_plot_batch)

        self.frame      = 4
        self.device     = torch.device('cuda')

        obs             = self.env.reset()
        obs             = prepro_half_one_dim(obs).unsqueeze(0)
        self.states     = torch.zeros(self.frame - 1, 80 * 80, device = self.device)
        self.states     = torch.concat((self.states, obs), dim = 0)

    def run(self):
        for _ in range(self.n_update):
            action      = self.agent.act(self.states)
            logprob     = self.agent.logprob(self.states, action)
            
            action_gym  = torch.where(
                    action != 0,
                    action + 1,
                    action
                )

            reward      = 0
            done        = False
            next_state  = None
            for i in range(self.frame):
                next_obs, reward_temp, done, _  = self.env.step(action_gym)
                next_obs                        = prepro_half_one_dim(next_obs).unsqueeze(0)

                reward      += reward_temp
                next_state  = next_obs if i == 0 else torch.concat((next_state, next_obs), dim = 0)

                if done:
                    if next_state.size(0) < self.frame:
                        next_obs      = torch.zeros(self.frame - next_state.size(0), 80 * 80, device = self.device)
                        next_state    = torch.concat((next_state, next_obs), dim = 0)
                    break 
            
            if self.is_save_memory:
                self.agent.save_obs(self.states, action, reward, done, next_state, logprob)
                
            self.states         = next_state
            self.eps_time       += 1 
            self.total_reward   += reward
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1
                print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, self.total_reward, self.eps_time))

                if self.i_episode % self.n_plot_batch == 0 and self.plotter is not None:
                    self.plotter.plot({
                        'Rewards': self.total_reward,
                        'Times': self.eps_time
                    })
                
                obs             = self.env.reset()
                obs             = prepro_half_one_dim(obs).unsqueeze(0)
                self.states     = torch.zeros(self.frame - 1, 80 * 80, device = self.device)
                self.states     = torch.concat((self.states, obs), dim = 0)

                self.total_reward   = 0
                self.eps_time       = 0

        return self.agent.get_obs(-self.n_update)
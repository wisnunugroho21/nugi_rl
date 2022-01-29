import torch

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.helpers.math import prepro_half_one_dim
from nugi_rl.train.runner.single_step.standard import SingleStepRunner
from nugi_rl.utilities.plotter.base import Plotter

class PongRunner(SingleStepRunner):
    def __init__(self, agent: Agent, env: Environment, is_save_memory: bool, render: bool, plotter: Plotter = None, n_plot_batch: int = 100) -> None:
        self.agent              = agent
        self.env                = env
        self.plotter            = plotter

        self.render             = render
        self.is_save_memory     = is_save_memory
        self.n_plot_batch       = n_plot_batch

        self.t_updates          = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0

        self.obs                = self.env.reset()  
        self.obs                = prepro_half_one_dim(self.obs)
        self.states             = self.obs

    def run(self) -> tuple:             
        action      = self.agent.act(self.states)
        logprob     = self.agent.logprob(self.states, action)

        action_gym  = torch.where(
                action != 0,
                action + 1,
                action
            )

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
            print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, self.total_reward, self.eps_time))

            if self.plotter is not None and self.i_episode % self.n_plot_batch == 0:
                self.plotter.plot({
                    'Rewards': self.total_reward,
                    'Times': self.eps_time
                })

            self.obs            = self.env.reset()  
            self.obs            = prepro_half_one_dim(self.obs)
            self.states         = self.obs

            self.total_reward   = 0
            self.eps_time       = 0

        return self.agent.get_obs(-1)
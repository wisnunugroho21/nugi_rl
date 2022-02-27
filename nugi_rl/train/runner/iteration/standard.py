from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.utilities.plotter.base import Plotter
from nugi_rl.train.runner.base import Runner

from copy import deepcopy

class IterRunner(Runner):
    def __init__(self, agent: Agent, env: Environment, is_save_memory: bool, render: bool, n_update: int, plotter: Plotter = None, n_plot_batch: int = 1) -> None:
        self.agent              = agent
        self.env                = env
        self.plotter            = plotter

        self.render             = render
        self.is_save_memory     = is_save_memory
        self.n_update           = n_update
        self.n_plot_batch       = n_plot_batch

        self.t_updates          = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0

        self.states             = self.env.reset()

    def run(self) -> tuple:
        for _ in range(self.n_update):
            action                      = self.agent.act(deepcopy(self.states))
            logprob                     = self.agent.logprob(deepcopy(self.states), action)

            next_state, reward, done, _ = self.env.step(action)
            
            if self.is_save_memory:
                self.agent.save_obs(deepcopy(self.states), action, reward, done, deepcopy(next_state), logprob)
                
            self.states         = next_state
            self.eps_time       += 1 
            self.total_reward   += reward
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1
                print('Episode {} \t t_reward: {} \t eps time: {}'.format(self.i_episode, self.total_reward, self.eps_time))

                if self.plotter is not None and self.i_episode % self.n_plot_batch == 0:
                    self.plotter.plot({
                        'Rewards': self.total_reward,
                        'Times': self.eps_time
                    })

                self.states         = self.env.reset()
                self.total_reward   = 0
                self.eps_time       = 0    

        return self.agent.get_obs(-self.n_update)
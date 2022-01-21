import ray

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.train.runner.iteration.standard import IterRunner
from nugi_rl.helpers.plotter.base import Plotter

@ray.remote(num_cpus = 4)
class SyncRunner(IterRunner):
    def __init__(self, agent: Agent, env: Environment, is_save_memory: bool, render: bool, n_update: int, plotter: Plotter = None, n_plot_batch: int = 100, tag: int = 1) -> None:
        super().__init__(agent, env, is_save_memory, render, n_update, plotter, n_plot_batch)

        self.states             = self.env.reset()
        self.tag                = tag        

    def run(self):
        self.agent.load_weights()
        self.agent.clear_obs()

        for _ in range(self.n_update):
            action                      = self.agent.act(self.states)
            logprob                     = self.agent.logprob(self.states, action)

            next_state, reward, done, _ = self.env.step(action)
            
            if self.is_save_memory:
                self.agent.save_obs(self.states, action, reward, done, next_state, logprob)
                
            self.states         = next_state
            self.eps_time       += 1 
            self.total_reward   += reward
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1

                if self.tag is not None:
                    print('Episode {} \t t_reward: {} \t time: {} \t tag: {}'.format(self.i_episode, self.total_reward, self.eps_time, self.tag))
                else:
                    print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, self.total_reward, self.eps_time))

                if self.plotter is not None and self.i_episode % self.n_plot_batch == 0:
                    self.plotter.plot({
                        'Rewards': self.total_reward,
                        'Times': self.eps_time
                    })

                self.states         = self.env.reset()
                self.total_reward   = 0
                self.eps_time       = 0

        if self.tag is not None:
            return self.agent.get_obs(-self.n_update), self.tag
        else:
            return self.agent.get_obs(-self.n_update)
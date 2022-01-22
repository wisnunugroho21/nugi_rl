from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.helpers.plotter.base import Plotter
from nugi_rl.train.runner.iteration.standard import IterRunner

class BraxIterRunner(IterRunner):
    def __init__(self, agent: Agent, env: Environment, is_save_memory: bool, render: bool, n_update: int, plotter: Plotter = None, n_plot_batch: int = 1) -> None:
        super().__init__(agent, env, is_save_memory, render, n_update, plotter, n_plot_batch)
        self.iter               = 0

    def run(self) -> tuple:
        states          = self.env.reset()
        total_reward    = 0

        for _ in range(self.n_update):
            action  = self.agent.act(states)
            logprob = self.agent.logprob(states, action)

            next_state, reward, done, _ = self.env.step(action)
            
            if self.is_save_memory:
                self.agent.save_all(states, action, reward, done, next_state, logprob)            
                           
            states          = next_state
            total_reward    += reward.mean()
                    
            if self.render:
                self.env.render()

        print('Iter {} \t t_reward: {}'.format(self.iter, total_reward))
        if self.plotter is not None and self.iter % self.n_plot_batch == 0:
            self.plotter.plot({
                'Rewards': total_reward
            }) 

        self.iter   += 1  

        return self.agent.get_obs(-self.n_update)
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.train.runner.base import Runner

class BraxIterRunner(Runner):
    def __init__(self, agent: Agent, env: Environment, is_save_memory: bool, render: bool, n_update: int, 
        writer: SummaryWriter = None, n_plot_batch: int = 100) -> None:

        self.agent              = agent
        self.env                = env

        self.render             = render
        self.is_save_memory     = is_save_memory
        self.n_update           = n_update
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch

        self.total_reward       = 0
        self.iter               = 0

        self.states             = self.env.reset()

    def run(self) -> tuple:
        for _ in range(self.n_update):
            action                      = self.agent.act(self.states)
            logprob                     = self.agent.logprob(self.states, action)

            next_state, reward, done, _ = self.env.step(action)
            
            if self.is_save_memory:
                self.agent.save_obs(self.states, action, reward, done, next_state, logprob)
                
            self.states         = next_state
            self.total_reward   += reward
                    
            if self.render:
                self.env.render()

        now = datetime.now()
        print('Iter {} \t t_reward: {} \t real time: {}'.format(self.iter, self.total_reward, now.strftime("%H:%M:%S")))

        self.states         = self.env.reset()
        self.total_reward   = 0        
        self.iter           += 1   

        return self.agent.get_obs(-self.n_update)
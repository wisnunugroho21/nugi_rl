from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.train.runner.base import Runner

class IterRunner(Runner):
    def __init__(self, agent: Agent, env: Environment, is_save_memory: bool, render: bool, n_update: int, 
        writer: SummaryWriter = None, n_plot_batch: int = 100) -> None:

        self.agent              = agent
        self.env                = env

        self.render             = render
        self.is_save_memory     = is_save_memory
        self.n_update           = n_update
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch

        self.t_updates          = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0

        self.states             = self.env.reset()

    def run(self) -> tuple:
        for _ in range(self.n_update):
            action                      = self.agent.act(self.states)
            next_state, reward, done, _ = self.env.step(action)
            
            if self.is_save_memory:
                self.agent.save_obs(self.states.tolist(), action, reward, float(done), next_state.tolist())
                
            self.states         = next_state
            self.eps_time       += 1 
            self.total_reward   += reward
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1
                now = datetime.now()

                print('Episode {} \t t_reward: {} \t eps time: {} \t real time: {}'.format(self.i_episode, self.total_reward, self.eps_time, now.strftime("%H:%M:%S")))

                if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                    self.writer.add_scalar('Rewards', self.total_reward, self.i_episode)
                    self.writer.add_scalar('Eps Time', self.eps_time, self.i_episode)

                self.states         = self.env.reset()
                self.total_reward   = 0
                self.eps_time       = 0    

        return self.agent.get_obs(-self.n_update)
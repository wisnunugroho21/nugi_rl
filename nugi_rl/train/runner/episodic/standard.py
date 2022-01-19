from torch.utils.tensorboard import SummaryWriter

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.train.runner.base import Runner

class EpisodicRunner(Runner):
    def __init__(self, agent: Agent, env: Environment, is_save_memory: bool, render: bool, n_update: int, 
        writer: SummaryWriter = None, n_plot_batch: int = 100) -> None:
        
        self.agent              = agent
        self.env                = env

        self.render             = render
        self.is_save_memory     = is_save_memory
        self.n_update           = n_update
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch

        self.i_episode          = 0

    def run(self) -> tuple:
        for _ in range(self.n_update):            
            state           = self.env.reset()
            done            = False
            total_reward    = 0
            eps_time        = 0

            while not done:
                action                      = self.agent.act(state)
                logprob                     = self.agent.logprob(state, action)

                next_state, reward, done, _ = self.env.step(action)
                
                if self.is_save_memory:
                    self.agent.save_obs(state, action, reward, done, next_state, logprob)
                    
                state           = next_state
                eps_time        += 1 
                total_reward    += reward
                        
                if self.render:
                    self.env.render()
                        
            self.i_episode  += 1
            print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, total_reward, eps_time))

            if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                self.writer.add_scalar('Rewards', total_reward, self.i_episode)
                self.writer.add_scalar('Times', eps_time, self.i_episode)

        return self.agent.get_obs(-eps_time)
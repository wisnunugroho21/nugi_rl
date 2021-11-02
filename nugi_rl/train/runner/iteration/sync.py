import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.train.runner.base import Runner

@ray.remote(num_gpus = 0.25)
class SyncRunner(Runner):
    def __init__(self, agent: Agent, env: Environment, is_save_memory: bool, render: bool, n_update: int, 
        writer: SummaryWriter = None, n_plot_batch: int = 100, tag: int = 1) -> None:
        
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

                if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                    self.writer.add_scalar('Rewards', self.total_reward, self.i_episode)
                    self.writer.add_scalar('Times', self.eps_time, self.i_episode)

                self.states         = self.env.reset()
                self.total_reward   = 0
                self.eps_time       = 0

        if self.tag is not None:
            return self.agent.get_obs(-self.n_update), self.tag
        else:
            return self.agent.get_obs(-self.n_update)
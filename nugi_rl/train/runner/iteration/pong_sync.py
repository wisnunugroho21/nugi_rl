from datetime import datetime
import ray

from torch.utils.tensorboard import SummaryWriter

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment

import torch
from nugi_rl.helpers.math import prepro_half_one_dim
from nugi_rl.train.runner.iteration.standard import IterRunner

@ray.remote(num_gpus = 0.25)
class PongSyncRunner(IterRunner):
    def __init__(self, agent: Agent, env: Environment, is_save_memory: bool, render: bool, n_update: int, writer: SummaryWriter = None, n_plot_batch: int = 100, tag: int = 1):
        super().__init__(agent, env, is_save_memory, render, n_update, writer, n_plot_batch)

        obs         = self.env.reset()  
        self.obs    = prepro_half_one_dim(obs)
        self.states = self.obs

        self.tag    = tag

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
                now = datetime.now()

                if self.tag is not None:
                    print('Episode {} \t t_reward: {} \t time: {} \t tag: {}'.format(self.i_episode, self.total_reward, self.eps_time, self.tag))
                else:
                    print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, self.total_reward, self.eps_time))

                if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                    self.writer.add_scalar('Rewards', self.total_reward, self.i_episode)
                    self.writer.add_scalar('Times', self.eps_time, self.i_episode)

                obs         = self.env.reset()  
                self.obs    = prepro_half_one_dim(obs)
                self.states = self.obs

                self.total_reward   = 0
                self.eps_time       = 0

        if self.tag is not None:
            return self.agent.get_obs(-self.n_update), self.tag
        else:
            return self.agent.get_obs(-self.n_update)
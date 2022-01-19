import torch
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.train.runner.base import Runner

class VectorizedRunner(Runner):
    def __init__(self, agent: Agent, envs: list[Environment], is_save_memory: bool, render: bool, n_update: int, 
        writer: SummaryWriter = None, n_plot_batch: int = 100) -> None:

        self.agent              = agent
        self.envs               = envs

        self.render             = render
        self.is_save_memory     = is_save_memory
        self.n_update           = n_update
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch

        self.t_updates          = [0 for _ in self.envs]
        self.i_episode          = [0 for _ in self.envs]
        self.total_reward       = [0 for _ in self.envs]
        self.eps_time           = [0 for _ in self.envs]

        self.states             = [env.reset() for env in self.envs]

    def run(self) -> tuple:
        for _ in range(self.n_update):
            actions                     = self.agent.act(torch.stack(self.states))
            logprobs                    = self.agent.logprob(torch.stack(self.states), actions)

            for index, (env, action, logprob) in enumerate(zip(self.envs, actions, logprobs)):
                next_state, reward, done, _ = env.step(action)
                
                if self.is_save_memory:
                    self.agent.save_obs(torch.stack(self.states), action, reward, done, next_state, logprob)
                    
                self.states[index]          = next_state
                self.eps_time[index]        += 1 
                self.total_reward[index]    += reward
                        
                if self.render:
                    env.render()

                if done:                
                    self.i_episode  += 1
                    now = datetime.now()

                    print('Agent {} Episode {} \t t_reward: {} \t eps time: {} \t real time: {}'.format(index, self.i_episode[index], self.total_reward[index], self.eps_time[index], now.strftime("%H:%M:%S")))

                    if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                        self.writer.add_scalar('Rewards', self.total_reward[index], self.i_episode[index])
                        self.writer.add_scalar('Eps Time', self.eps_time[index], self.i_episode[index])

                    self.states         = env.reset()
                    self.total_reward   = 0
                    self.eps_time       = 0    

        return self.agent.get_obs(-1 * self.n_update * len(self.envs))
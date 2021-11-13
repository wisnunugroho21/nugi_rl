from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from nugi_rl.agent.base import Agent
from nugi_rl.environment.base import Environment
from nugi_rl.train.runner.base import Runner

class BraxIterRunner(Runner):
    def __init__(self, agent: Agent, env: Environment, is_save_memory: bool, render: bool, n_update: int, 
        writer: SummaryWriter = None) -> None:

        self.agent              = agent
        self.env                = env

        self.render             = render
        self.is_save_memory     = is_save_memory
        self.n_update           = n_update
        self.writer             = writer

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

            if self.writer is not None:
                self.writer.add_scalar('Rewards', total_reward, self.iter)
                           
            states          = next_state
            total_reward    += reward.mean()
                    
            if self.render:
                self.env.render()

        print('Iter {} \t t_reward: {} \t real time: {}'.format(self.iter, total_reward, datetime.now().strftime("%H:%M:%S")))     
        self.iter   += 1   

        return self.agent.get_obs(-self.n_update)
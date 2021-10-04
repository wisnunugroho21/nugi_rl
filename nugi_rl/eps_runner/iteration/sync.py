import numpy as np
import torch

from eps_runner.iteration.iter_runner import IterRunner
import ray

@ray.remote(num_gpus = 0.25)
class SyncRunner(IterRunner):
    def __init__(self, agent, env, is_save_memory, render, n_update, is_discrete, max_action, writer , n_plot_batch, tag = None):
        self.agent              = agent
        self.env                = env

        self.render             = render
        self.is_save_memory     = is_save_memory
        self.n_update           = n_update
        self.max_action         = max_action
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch
        self.is_discrete        = is_discrete
        self.tag                = tag        

        self.t_updates          = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0

        self.states             = self.env.reset()

    def run(self):
        self.agent.load_weights(device = torch.device('cpu'))
        self.agent.memory.clear_memory()

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
            return self.agent.memory.get_ranged_items(-self.n_update), self.tag
        else:
            return self.agent.memory.get_ranged_items(-self.n_update)
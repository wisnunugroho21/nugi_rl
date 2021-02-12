import numpy as np

from utils.math_function import plot
# from torch.utils.tensorboard import SummaryWriter
from memory.list_memory import ListMemory

import time
import datetime

import ray

class SyncExecutor():
    def __init__(self, agent, envs, n_iteration, Runner, reward_threshold, save_weights = False, n_plot_batch = 100, render = True, training_mode = True, n_update = 1024, n_aux_update = 10, 
        n_saved = 10, max_action = 1.0, reward_target = 300):

        self.agent              = agent
        self.n_iteration        = n_iteration
        self.save_weights       = save_weights
        self.n_saved            = n_saved
        self.reward_threshold   = reward_threshold
        self.n_plot_batch       = n_plot_batch
        self.max_action         = max_action
        self.n_aux_update       = n_aux_update
        self.reward_target      = reward_target

        self.t_updates          = 0
        self.t_aux_updates      = 0

        ray.init()
        self.runners            = [Runner.remote(env, render, training_mode, n_update, agent = agent, max_action = max_action, n_plot_batch = n_plot_batch) for env in envs]
        
    def execute_discrete(self):
        start = time.time()
        print('Running the training!!')
        
        try:            
            for i_iteration in range(self.n_iteration):
                self.agent.save_temp_weights()
                futures  = [runner.run_discrete_iteration.remote() for runner in self.runners]
                memories = ray.get(futures)

                for memory in memories:
                    self.agent.save_memory(memory)

                self.agent.update_ppo()
                self.t_aux_updates += 1                

                if self.t_aux_updates == self.n_aux_update:
                    self.agent.update_aux()
                    self.t_aux_updates = 0   

                if self.save_weights:
                    if i_iteration % self.n_saved == 0:
                        self.agent.save_weights() 
                        print('weights saved')

        finally:
            ray.shutdown()
            
            finish = time.time()
            timedelta = finish - start
            print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))

    def execute_continous(self):
        start = time.time()
        print('Running the training!!')

        try:
            for i_iteration in range(self.n_iteration):
                self.agent.save_temp_weights()
                futures  = [runner.run_continous_iteration.remote() for runner in self.runners]
                memories = ray.get(futures)

                for memory in memories:
                    self.agent.save_memory(memory)

                self.agent.update_ppo()
                self.t_aux_updates += 1                

                if self.t_aux_updates == self.n_aux_update:
                    self.agent.update_aux()
                    self.t_aux_updates = 0

                if self.save_weights:
                    if i_iteration % self.n_saved == 0:
                        self.agent.save_weights()
                        print('weights saved')

        finally:
            ray.shutdown()
            
            finish = time.time()
            timedelta = finish - start
            print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))
            
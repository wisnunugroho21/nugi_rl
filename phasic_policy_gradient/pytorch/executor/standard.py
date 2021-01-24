import numpy as np
from torch.utils.tensorboard import SummaryWriter

import time
import datetime

from memory.list_memory import ListMemory

class StandardExecutor():
    def __init__(self, agent, env, n_iteration, Runner, reward_threshold, save_weights = False, n_plot_batch = 100, render = True, training_mode = True, n_update = 1024, n_aux_update = 10, 
        n_saved = 10, max_action = 1.0):

        self.agent              = agent
        self.runner             = Runner(env, render, training_mode, n_update, max_action = max_action, writer = SummaryWriter(), n_plot_batch = n_plot_batch)

        self.n_iteration        = n_iteration
        self.save_weights       = save_weights
        self.n_saved            = n_saved
        self.reward_threshold   = reward_threshold
        self.n_plot_batch       = n_plot_batch
        self.max_action         = max_action
        self.n_aux_update       = n_aux_update

        self.t_updates          = 0
        self.t_aux_updates      = 0  
        
    def execute_discrete(self):
        start = time.time()
        print('Running the training!!')

        try:
            for i_iteration in range(self.n_iteration):
                memories  = self.runner.run_discrete_iteration(self.agent)
                self.agent.save_memory(memories)

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
            finish = time.time()
            timedelta = finish - start
            print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))

    def execute_continous(self):
        start = time.time()
        print('Running the training!!')

        try:
            for i_iteration in range(self.n_iteration):                
                memories = self.runner.run_continous_iteration(self.agent)
                self.agent.save_memory(memories)

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
            finish = time.time()
            timedelta = finish - start
            print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))
            
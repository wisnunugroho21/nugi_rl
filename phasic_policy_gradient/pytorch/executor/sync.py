import numpy as np
import matplotlib.pyplot as plt

from utils.math_function import plot
from torch.utils.tensorboard import SummaryWriter
from eps_runner.vectorized_eps import VectorizedRunner
from memory.list_memory import ListMemory

import time
import datetime

class VectorizedExecutor():
    def __init__(self, agent, env, n_iteration, Runner, reward_threshold, save_weights = False, n_plot_batch = 100, render = True, training_mode = True, n_update = 1024, n_aux_update = 10, 
        n_saved = 10, params_max = 1.0, params_min = 0.2, params_subtract = 0.0001, params_dynamic = True, max_action = 1.0):

        self.agent              = agent
        self.runner             = VectorizedRunner(env, render, training_mode, n_update, max_action = max_action, writer = SummaryWriter(), n_plot_batch = n_plot_batch)
        self.memories           = [ListMemory() for _ in range(len(env))]

        self.n_iteration        = n_iteration
        self.save_weights       = save_weights
        self.n_saved            = n_saved
        self.reward_threshold   = reward_threshold
        self.n_plot_batch       = n_plot_batch
        self.max_action         = max_action
        self.n_aux_update       = n_aux_update

        self.params_min         = params_min
        self.params_subtract    = params_subtract
        self.params_dynamic     = params_dynamic
        self.params             = params_max 

        self.t_updates          = 0
        self.t_aux_updates      = 0
        
    def execute_discrete(self):
        start = time.time()
        print('Running the training!!')

        try:
            for i_iteration in range(self.n_iteration):
                self.agent, self.memories  = self.runner.run_discrete_iteration(self.agent, self.memories)

                for memory in self.memories:
                    temp_states, temp_actions, temp_rewards, temp_dones, temp_next_states = memory.get_all_items()
                    self.agent.save_all(temp_states, temp_actions, temp_rewards, temp_dones, temp_next_states)
                    memory.clear_memory()

                self.agent.update_ppo()
                self.t_aux_updates += 1                

                if self.t_aux_updates == self.n_aux_update:
                    self.agent.update_aux()
                    self.t_aux_updates = 0

                if self.params_dynamic:
                    self.params = self.params - self.params_subtract
                    self.params = self.params if self.params > self.params_min else self.params_min      

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
                self.agent, self.memories  = self.runner.run_continous_iteration(self.agent, self.memories)

                for memory in self.memories:
                    temp_states, temp_actions, temp_rewards, temp_dones, temp_next_states = memory.get_all_items()
                    self.agent.save_all(temp_states, temp_actions, temp_rewards, temp_dones, temp_next_states)
                    memory.clear_memory()

                self.agent.update_ppo()
                self.t_aux_updates += 1                

                if self.t_aux_updates == self.n_aux_update:
                    self.agent.update_aux()
                    self.t_aux_updates = 0

                if self.params_dynamic:
                    self.params = self.params - self.params_subtract
                    self.params = self.params if self.params > self.params_min else self.params_min

                if self.save_weights:
                    if i_iteration % self.n_saved == 0:
                        self.agent.save_weights()
                        print('weights saved')

        finally:
            finish = time.time()
            timedelta = finish - start
            print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))
            
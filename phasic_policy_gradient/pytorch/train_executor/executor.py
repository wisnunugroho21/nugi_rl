import numpy as np
import time
import datetime

class Executor():
    def __init__(self, agent, env, n_iteration, runner, reward_threshold, save_weights = False, n_plot_batch = 100, render = True, training_mode = True, n_update = 1024, n_aux_update = 10, 
        n_saved = 10, max_action = 1.0, load_weights = False):

        self.agent              = agent
        self.env                = env
        self.runner             = runner

        self.n_iteration        = n_iteration
        self.save_weights       = save_weights
        self.n_saved            = n_saved
        self.reward_threshold   = reward_threshold
        self.n_plot_batch       = n_plot_batch
        self.max_action         = max_action
        self.n_aux_update       = n_aux_update

        self.t_updates          = 0
        self.t_aux_updates      = 0

        if load_weights:
            self.agent.load_weights()
            print('Weight Loaded')  

    def execute(self):
        start = time.time()
        print('Running the training!!')

        try:
            for i_iteration in range(self.n_iteration):
                memories  = self.runner.run_iteration(self.agent)
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
            print('\nTimelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))
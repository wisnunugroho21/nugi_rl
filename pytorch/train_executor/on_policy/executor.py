import numpy as np
import time
import datetime

class OnExecutor():
    def __init__(self, agent, env, n_iteration, runner, reward_threshold, save_weights = False, n_plot_batch = 100,
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

                self.agent.update()

                if self.save_weights:
                    if i_iteration % self.n_saved == 0:
                        self.agent.save_weights()
                        print('weights saved')

        finally:
            finish = time.time()
            timedelta = finish - start
            print('\nTimelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))
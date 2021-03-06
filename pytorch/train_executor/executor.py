import numpy as np
import time
import datetime

class Executor():
    def __init__(self, agent, n_iteration, runner, save_weights = False, n_saved = 10, load_weights = False):
        self.agent              = agent
        self.runner             = runner

        self.n_iteration        = n_iteration
        self.save_weights       = save_weights
        self.n_saved            = n_saved

        if load_weights:
            self.agent.load_weights()
            print('Weight Loaded')  

    def execute(self):
        start = time.time()
        print('Running the training!!')

        try:
            for i_iteration in range(1, self.n_iteration, 1):
                memories  = self.runner.run()

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
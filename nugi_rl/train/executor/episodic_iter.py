import time
import datetime

from nugi_rl.train.runner.base import Runner
from nugi_rl.train.executor.base import Executor
from nugi_rl.agent.base import Agent

class Executor(Executor):
    def __init__(self, agent: Agent, n_iteration: int, n_episodic: int, runner: Runner, save_weights: bool = False, 
        n_saved: int = 10, load_weights: bool = False, is_training_mode: bool = True):

        self.agent              = agent
        self.runner             = runner

        self.n_iteration        = n_iteration
        self.n_episodic         = n_episodic

        self.save_weights       = save_weights
        self.n_saved            = n_saved
        self.is_training_mode   = is_training_mode 
        self.load_weights       = load_weights

        self.i_episodic         = 1       

    def execute(self):
        if self.load_weights:
            self.agent.load_weights()
            print('Weight Loaded')  

        start = time.time()
        print('Running the training!!')

        try:
            for i_iteration in range(1, self.n_iteration, 1):
                _, _, _, dones, _ = self.runner.run()                

                if self.is_training_mode:
                    if dones[-1] == 1:
                        if self.i_episodic % self.n_episodic == 0:
                            self.agent.update('episodic')                         
                        self.i_episodic += 1

                    if i_iteration % self.n_iteration == 0:
                        self.agent.update('iter')
                        
                    if self.save_weights:
                        if i_iteration % self.n_saved == 0:
                            self.agent.save_weights()
                            print('weights saved')

        except KeyboardInterrupt:
            print('Stopped by User')
        finally:
            finish = time.time()
            timedelta = finish - start
            print('\nTimelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))
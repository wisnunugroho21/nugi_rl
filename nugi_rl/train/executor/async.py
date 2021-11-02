import time
import datetime
import ray

from nugi_rl.train.runner.base import Runner
from nugi_rl.train.executor.base import Executor
from nugi_rl.agent.base import Agent

class SyncExecutor(Executor):
    def __init__(self, agent: Agent, n_iteration: int, runner: Runner, save_weights: bool = False, 
        n_saved: int = 10, load_weights: bool = False, is_training_mode: bool = True):

        self.agent              = agent
        self.runner             = runner

        self.n_iteration        = n_iteration
        self.save_weights       = save_weights
        self.n_saved            = n_saved
        self.is_training_mode   = is_training_mode 
        self.load_weights       = load_weights
        
    def execute(self):
        if self.load_weights:
            self.agent.load_weights()
            print('Weight Loaded')

        start = time.time()
        print('Running the training!!')
        
        try:      
            self.agent.save_weights()

            futures = []
            for runner in enumerate(self.runner):
                futures.append(runner.run.remote())
                time.sleep(2)

            for _ in range(self.n_iteration):
                ready, not_ready = ray.wait(futures)
                memory, tag = ray.get(ready)[0]

                futures = not_ready
                futures.append(self.runner[tag].run.remote())

                if self.is_training_mode:
                    self.agent.save_memory(memory)
                    self.agent.update()
                self.agent.save_weights()

        except KeyboardInterrupt:
            print('Stopped by User')
        finally:
            ray.shutdown()
            
            finish = time.time()
            timedelta = finish - start
            print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))
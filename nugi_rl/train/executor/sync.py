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
            for _ in range(self.n_iteration):
                self.agent.save_weights()

                futures = [runner.run.remote() for runner in self.runner]
                datas   = ray.get(futures)

                if self.is_training_mode:
                    for data in datas:
                        memory, _ = data

                        states, actions, rewards, dones, next_states = memory
                        self.agent.save_all(states, actions, rewards, dones, next_states)
                
                    self.agent.update()

        except KeyboardInterrupt:
            print('Stopped by User')
        finally:
            ray.shutdown()
            
            finish = time.time()
            timedelta = finish - start
            print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))
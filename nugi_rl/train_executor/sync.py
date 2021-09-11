import time
import datetime

import ray

class SyncExecutor():
    def __init__(self, agent, n_iteration, runner, save_weights = False, n_saved = 10, load_weights = False, is_training_mode = True):
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

                futures  = [runner.run.remote() for runner in self.runner]
                memories = ray.get(futures)

                for memory in memories:
                    states, actions, rewards, dones, next_states = memory
                    self.agent.memory.save_all(states, actions, rewards, dones, next_states)

                self.agent.update()

        except KeyboardInterrupt:
            print('Stopped by User')
        finally:
            ray.shutdown()
            
            finish = time.time()
            timedelta = finish - start
            print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))
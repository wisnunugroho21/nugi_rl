import numpy as np

from utils.math_function import plot
# from torch.utils.tensorboard import SummaryWriter

import time
import datetime

import ray

class SyncExecutor():
    def __init__(self, agents, agent_learner, folder_agents, n_iteration, runners, save_weights = False, n_saved = 10, load_weights = False):

        self.agents             = agents
        self.agent_learner      = agent_learner
        self.runners            = runners
        
        self.n_iteration        = n_iteration
        self.save_weights       = save_weights
        self.n_saved            = n_saved
        self.folder_agents      = folder_agents

        self.t_updates          = 0
        self.t_aux_updates      = 0
                
    def execute(self):
        start = time.time()
        print('Running the training!!')
        
        try:
            episode_ids = []
            for runner in self.runners:
                episode_ids.append(runner.run_iteration.remote())
                time.sleep(4)

            for i_iteration in range(self.n_iteration):
                self.agent_learner.save_weights()

                ready, not_ready    = ray.wait(episode_ids)
                memory, tag         = ray.get(ready)[0]

                self.agents[tag].save_on_memory(memory)
                self.agents[tag].update_on()

                episode_ids = not_ready
                episode_ids.append(self.runners[tag].run_iteration.remote()) 

                self.agent_learner.save_off_memory(memory)
                self.agent_learner.update_off()

                if self.save_weights:
                    if i_iteration % self.n_saved == 0:
                        self.agent.save_weights() 
                        print('weights saved')

        finally:
            ray.shutdown()
            
            finish = time.time()
            timedelta = finish - start
            print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))
import time
import datetime

class EpisodicIterExecutor():
    def __init__(self, agent, n_iteration, n_update_iter, n_update_episodic, runner, save_weights = False, n_saved = 10, load_weights = False, is_training_mode = True):
        self.agent              = agent
        self.runner             = runner

        self.n_iteration        = n_iteration
        self.n_update_episodic  = n_update_episodic
        self.n_update_iter      = n_update_iter 
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
                    if dones[0] == 1:
                        if self.i_episodic % self.n_update_episodic == 0:
                            self.agent.update('episodic')                         
                        self.i_episodic += 1

                    if i_iteration % self.n_update_iter == 0:
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
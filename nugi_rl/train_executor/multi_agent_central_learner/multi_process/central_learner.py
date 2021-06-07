import datetime
import time

class CentralLearnerExecutor():
    def __init__(self, agent, n_iteration, memory, runner, save_weights = False, n_saved = 10):
        self.agent  = agent
        self.memory = memory
        self.runner = runner

        self.n_iteration    = n_iteration
        self.save_weights   = save_weights
        self.n_saved        = n_saved

    def execute(self):
        start = time.time()
        print('Running the training!!')

        try:
            for i_iteration in range(1, self.n_iteration, 1):
                if self.memory.check_if_exists_redis():
                    self.memory.load_redis()
                    self.agent.save_memory(self.memory)

                self.runner.run()
                self.agent.update()

                if self.save_weights:
                    if i_iteration % self.n_saved == 0:
                        self.agent.save_weights()
                        print('weights saved')

        except KeyboardInterrupt:
            print('Stopped by User')
        finally:
            finish = time.time()
            timedelta = finish - start
            print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))
            
from copy import deepcopy

class RedisIterWrapRunner():
    def __init__(self, agent, runner, n_update):
        self.agent      = agent
        self.n_update   = n_update
        self.runner     = runner

    def run(self):        
        for i in range(1, self.n_update, 1):
            self.runner.run()            
            self.agent.memory.save_redis(-1)

        return self.agent.memory.get_ranged_items(-self.n_update)
class RedisIterWrapRunner():
    def __init__(self, runner, memory, n_update):
        self.n_update   = n_update
        self.memories   = memory
        self.runner     = runner

    def run(self):
        for i in range(1, self.n_update, 1):
            memories  = self.runner.run()            
            memories.save_redis()

            states, actions, rewards, dones, next_states = memories.get_all_items()
            self.memories.save_all(states, actions, rewards, dones, next_states)

        return self.memories
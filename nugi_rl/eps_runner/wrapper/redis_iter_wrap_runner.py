class RedisIterWrapRunner():
    def __init__(self, runner, memory, n_update):
        self.n_update   = n_update
        self.memories   = memory
        self.runner     = runner

    def run(self):
        for _ in range(1, self.n_update, 1):
            memories  = self.runner.run()
            
            memories.save_redis()
            self.memories.save_memory(memories)

        return self.memories
from memory.policy.standard import PolicyMemory
import json

class PolicyRedisListMemory(PolicyMemory):
    def __init__(self, redis, capacity = 100000, n_update = 1024, datas = None):
        super().__init__(capacity, datas)
        self.redis      = redis
        self.n_update   = n_update

    def save_redis(self):
        for state, action, reward, done, next_state in zip(self.states, self.actions, self.rewards, self.dones, self.next_states):
            self.redis.rpush('states', json.dumps(state))
            self.redis.rpush('actions', json.dumps(action))
            self.redis.rpush('rewards', json.dumps(reward))
            self.redis.rpush('dones', json.dumps(done))
            self.redis.rpush('next_states', json.dumps(next_state))

    def load_redis(self):
        states         = list(map(lambda e: json.loads(e), self.redis.lrange('states', 0, -1)))
        actions        = list(map(lambda e: json.loads(e), self.redis.lrange('actions', 0, -1)))
        rewards        = list(map(lambda e: json.loads(e), self.redis.lrange('rewards', 0, -1)))
        dones          = list(map(lambda e: json.loads(e), self.redis.lrange('dones', 0, -1)))
        next_states    = list(map(lambda e: json.loads(e), self.redis.lrange('next_states', 0, -1)))

        self.save_all(self, states, actions, rewards, dones, next_states)

    def delete_redis(self):
        self.redis.delete('states')
        self.redis.delete('actions')
        self.redis.delete('rewards')
        self.redis.delete('dones')
        self.redis.delete('next_states')

    def check_if_exists_redis(self):
        return bool(self.redis.exists('dones'))
        
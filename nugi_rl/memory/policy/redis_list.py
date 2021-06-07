from memory.policy.standard import PolicyMemory
import json

class PolicyRedisListMemory(PolicyMemory):
    def __init__(self, redis, capacity = 100000, datas = None):
        super().__init__(capacity, datas)
        self.redis = redis

    def save_redis(self):
        for state, action, reward, done, next_state in zip(self.states, self.actions, self.rewards, self.dones, self.next_states):
            self.redis.rpush('states', json.dumps(state))
            self.redis.rpush('actions', json.dumps(action))
            self.redis.rpush('rewards', json.dumps(reward))
            self.redis.rpush('dones', json.dumps(done))
            self.redis.rpush('next_states', json.dumps(next_state))

    def load_redis(self):
        self.states         = list(map(lambda e: json.loads(e), self.redis.lrange('states', 0, -1)))
        self.actions        = list(map(lambda e: json.loads(e), self.redis.lrange('actions', 0, -1)))
        self.rewards        = list(map(lambda e: json.loads(e), self.redis.lrange('rewards', 0, -1)))
        self.dones          = list(map(lambda e: json.loads(e), self.redis.lrange('dones', 0, -1)))
        self.next_states    = list(map(lambda e: json.loads(e), self.redis.lrange('next_states', 0, -1)))

    def delete_redis(self):
        self.redis.delete('states')
        self.redis.delete('actions')
        self.redis.delete('rewards')
        self.redis.delete('dones')
        self.redis.delete('next_states')

    def check_if_exists_redis(self):
        if bool(self.redis.exists('dones')):
            if self.redis.llen('dones') > 1024:
                return True

        return False
        
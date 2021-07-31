from memory.policy.standard import PolicyMemory
import json

class PolicyRedisListMemory(PolicyMemory):
    def __init__(self, redis, capacity = 100000, datas = None):
        super().__init__(capacity, datas)
        self.redis      = redis

    def save_redis(self, start_position = 0, end_position = None):
        if end_position is not None or end_position == -1:
            states      = self.states[start_position:end_position + 1]
            actions     = self.actions[start_position:end_position + 1]
            rewards     = self.rewards[start_position:end_position + 1]
            dones       = self.dones[start_position:end_position + 1]
            next_states = self.next_states[start_position:end_position + 1]
        else:
            states      = self.states[start_position:]
            actions     = self.actions[start_position:]
            rewards     = self.rewards[start_position:]
            dones       = self.dones[start_position:]
            next_states = self.next_states[start_position:]

        for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
            self.redis.rpush('states', json.dumps(state))
            self.redis.rpush('actions', json.dumps(action))
            self.redis.rpush('rewards', json.dumps(reward))
            self.redis.rpush('dones', json.dumps(done))
            self.redis.rpush('next_states', json.dumps(next_state))

    def load_redis(self, start_position = 0, end_position = -1):
        states         = list(map(lambda e: json.loads(e), self.redis.lrange('states', start_position, end_position)))
        actions        = list(map(lambda e: json.loads(e), self.redis.lrange('actions', start_position, end_position)))
        rewards        = list(map(lambda e: json.loads(e), self.redis.lrange('rewards', start_position, end_position)))
        dones          = list(map(lambda e: json.loads(e), self.redis.lrange('dones', start_position, end_position)))
        next_states    = list(map(lambda e: json.loads(e), self.redis.lrange('next_states', start_position, end_position)))

        self.save_all(states, actions, rewards, dones, next_states)

    def delete_redis(self):
        self.redis.delete('states')
        self.redis.delete('actions')
        self.redis.delete('rewards')
        self.redis.delete('dones')
        self.redis.delete('next_states')

    def check_if_exists_redis(self):
        return bool(self.redis.exists('dones'))
        
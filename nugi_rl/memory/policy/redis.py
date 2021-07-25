from copy import deepcopy
import torch
import json

from memory.policy.standard import PolicyMemory

class RedisPolicyMemory(PolicyMemory):
    def __init__(self, redis, capacity = 100000):
        self.redis          = redis
        self.capacity       = capacity
        self.position       = 0      

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        states         = list(map(lambda e: json.loads(e), self.redis.lrange('states', idx, idx + 1)))
        actions        = list(map(lambda e: json.loads(e), self.redis.lrange('actions', idx, idx + 1)))
        rewards        = list(map(lambda e: json.loads(e), self.redis.lrange('rewards', idx, idx + 1)))
        dones          = list(map(lambda e: json.loads(e), self.redis.lrange('dones', idx, idx + 1)))
        next_states    = list(map(lambda e: json.loads(e), self.redis.lrange('next_states', idx, idx + 1)))

        return torch.FloatTensor(states), torch.FloatTensor(actions), torch.FloatTensor(rewards), \
            torch.FloatTensor(dones), torch.FloatTensor(next_states)

    def save_eps(self, state, action, reward, done, next_state):
        if len(self) >= self.capacity:
            self.redis.ltrim('states', 0, -2)
            self.redis.ltrim('actions', 0, -2)
            self.redis.ltrim('rewards', 0, -2)
            self.redis.ltrim('dones', 0, -2)
            self.redis.ltrim('next_states', 0, -2)

        self.redis.rpush('states', json.dumps(state))
        self.redis.rpush('actions', json.dumps(action))
        self.redis.rpush('rewards', json.dumps(reward))
        self.redis.rpush('dones', json.dumps(done))
        self.redis.rpush('next_states', json.dumps(next_state))

    def save_replace_all(self, states, actions, rewards, dones, next_states):
        self.clear_memory()
        self.save_all(states, actions, rewards, dones, next_states)

    def save_all(self, states, actions, rewards, dones, next_states):
        for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
            self.save_eps(state, action, reward, done, next_state)

    def get_all_items(self):         
        return self.states, self.actions, self.rewards, self.dones, self.next_states 

    def clear_memory(self):
        self.redis.delete('states')
        self.redis.delete('actions')
        self.redis.delete('rewards')
        self.redis.delete('dones')
        self.redis.delete('next_states')

    def clear_idx(self, idx):
        self.redis.ltrim('states', idx, idx + 1)
        self.redis.ltrim('actions', idx, idx + 1)
        self.redis.ltrim('rewards', idx, idx + 1)
        self.redis.ltrim('dones', idx, idx + 1)
        self.redis.ltrim('next_states', idx, idx + 1)
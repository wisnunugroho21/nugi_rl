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
        return self.redis.llen('dones')

    def __getitem__(self, idx):
        idx = idx.item()

        states         = list(map(lambda e: json.loads(e), self.redis.lrange('states', idx, idx)))
        actions        = list(map(lambda e: json.loads(e), self.redis.lrange('actions', idx, idx)))
        rewards        = list(map(lambda e: json.loads(e), self.redis.lrange('rewards', idx, idx)))
        dones          = list(map(lambda e: json.loads(e), self.redis.lrange('dones', idx, idx)))
        next_states    = list(map(lambda e: json.loads(e), self.redis.lrange('next_states', idx, idx)))

        return torch.tensor(states, dtype = torch.float32), torch.tensor(actions, dtype = torch.float32), \
            torch.tensor(rewards, dtype = torch.float32), torch.tensor(dones, dtype = torch.float32), \
            torch.tensor(next_states, dtype = torch.float32)

    def save_obs(self, state, action, reward, done, next_state):
        if len(self) >= self.capacity:
            self.redis.ltrim('states', 1, -1)
            self.redis.ltrim('actions', 1, -1)
            self.redis.ltrim('rewards', 1, -1)
            self.redis.ltrim('dones', 1, -1)
            self.redis.ltrim('next_states', 1, -1)

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
            self.save_obs(state, action, reward, done, next_state)

    def get_all_items(self):         
        return self.states, self.actions, self.rewards, self.dones, self.next_states

    def get_ranged_items(self, start_position = 0, end_position = -1):   
        states         = list(map(lambda e: json.loads(e), self.redis.lrange('states', start_position, end_position)))
        actions        = list(map(lambda e: json.loads(e), self.redis.lrange('actions', start_position, end_position)))
        rewards        = list(map(lambda e: json.loads(e), self.redis.lrange('rewards', start_position, end_position)))
        dones          = list(map(lambda e: json.loads(e), self.redis.lrange('dones', start_position, end_position)))
        next_states    = list(map(lambda e: json.loads(e), self.redis.lrange('next_states', start_position, end_position)))

        return states, actions, rewards, dones, next_states  

    def clear_memory(self):
        self.redis.delete('states')
        self.redis.delete('actions')
        self.redis.delete('rewards')
        self.redis.delete('dones')
        self.redis.delete('next_states')

    def clear_idx(self, idx):
        raise Exception('not yet implemented! need more works for this function')
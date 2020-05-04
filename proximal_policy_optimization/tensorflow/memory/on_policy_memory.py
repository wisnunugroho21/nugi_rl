import tensorflow as tf

class OnMemory():
    def __init__(self):
        self.actions = [] 
        self.states = []
        self.rewards = []
        self.dones = []     
        self.next_states = []        

    def save_eps(self, state, reward, action, done, next_state):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)
        self.next_states.append(next_state) 

    def save_replace_all(self, states, rewards, actions, dones, next_states):
        self.rewards = rewards
        self.states = states
        self.actions = actions
        self.dones = dones
        self.next_states = next_states

    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]        

    def length(self):
        return len(self.dones)

    def get_dataset_items(self):         
        states = tf.constant(self.states, dtype = tf.float32)
        actions = tf.constant(self.actions, dtype = tf.float32) 
        rewards = tf.expand_dims(tf.constant(self.rewards, dtype = tf.float32), 1)
        dones = tf.expand_dims(tf.constant(self.dones, dtype = tf.float32), 1)
        next_states = tf.constant(self.next_states, dtype = tf.float32) 
        
        return tf.data.Dataset.from_tensor_slices((states, actions, rewards, dones, next_states))

    def get_all_items(self):         
        return self.states, self.rewards, self.actions, self.dones, self.next_states 

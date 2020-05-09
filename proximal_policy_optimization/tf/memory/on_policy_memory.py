import tensorflow as tf

class OnMemory:
    def __init__(self, datas = None):
        if datas is None :
            self.states         = []
            self.actions        = []
            self.rewards        = []
            self.dones          = []
            self.next_states    = []
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states = datas

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx, to_tensor = True):
        if to_tensor:
            return tf.constant(self.states[idx], dtype = tf.float32), tf.constant(self.actions[idx], dtype = tf.float32), \
                tf.constant([self.rewards[idx]], dtype = tf.float32), tf.constant([self.dones[idx]], dtype = tf.float32), \
                tf.constant(self.next_states[idx], dtype = tf.float32)

        else:
            return self.states[idx], self.actions[idx], [self.rewards[idx]], [self.dones[idx]], self.next_states[idx]

    def get_all_items(self, to_tensor_dataset = True):
        if to_tensor_dataset:
            states = tf.constant(self.states, dtype = tf.float32)
            actions = tf.constant(self.actions, dtype = tf.float32) 
            rewards = tf.expand_dims(tf.constant(self.rewards, dtype = tf.float32), 1)
            dones = tf.expand_dims(tf.constant(self.dones, dtype = tf.float32), 1)
            next_states = tf.constant(self.next_states, dtype = tf.float32)        

            return tf.data.Dataset.from_tensor_slices((states, actions, rewards, dones, next_states)) 

        else:  
            return self.states, self.actions, self.rewards, self.dones, self.next_states

    def save_eps(self, state, action, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)            

    def save_replace_all(self, states, actions, rewards, dones, next_states):
        self.states         = states
        self.actions        = actions
        self.rewards        = rewards
        self.dones          = dones
        self.next_states    = next_states    

    def clearMemory(self, idx = -100):
        if idx == -100:
            del self.states[:]
            del self.actions[:]
            del self.rewards[:]
            del self.dones[:]
            del self.next_states[:]

        else:
            del self.states[idx]
            del self.actions[idx]
            del self.rewards[idx]
            del self.dones[idx]
            del self.next_states[idx]

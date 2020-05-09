import tensorflow as tf

from on_policy_memory import OnMemory

class OnMemoryImpala(OnMemory):
    def __init__(self, datas = None):
        super().__init__(datas)

        if datas is None :
            self.worker_action_datas = []
        else:
            self.worker_action_datas = datas[-1]

    def __getitem__(self, idx, to_tensor = True):
        if to_tensor:
            return tf.constant(self.states[idx], dtype = tf.float32), tf.constant(self.actions[idx], dtype = tf.float32), \
                tf.constant([self.rewards[idx]], dtype = tf.float32), tf.constant([self.dones[idx]], dtype = tf.float32), \
                tf.constant(self.next_states[idx], dtype = tf.float32), tf.constant(self.worker_action_datas[idx], dtype = tf.float32)

        else:
            return self.states[idx], self.actions[idx], [self.rewards[idx]], [self.dones[idx]], self.next_states[idx], self.worker_action_datas[idx]

    def get_all_items(self, to_tensor_dataset = True):
        if to_tensor_dataset:
            states              = tf.constant(self.states, dtype = tf.float32)
            actions             = tf.constant(self.actions, dtype = tf.float32) 
            rewards             = tf.expand_dims(tf.constant(self.rewards, dtype = tf.float32), 1)
            dones               = tf.expand_dims(tf.constant(self.dones, dtype = tf.float32), 1)
            next_states         = tf.constant(self.next_states, dtype = tf.float32)       
            worker_action_datas = tf.constant(self.worker_action_datas, dtype = tf.float32)

            return tf.data.Dataset.from_tensor_slices((states, actions, rewards, dones, next_states, worker_action_datas)) 

        else:  
            return self.states, self.actions, self.rewards, self.dones, self.next_states, self.worker_action_datas

    def save_eps(self, state, reward, action, done, next_state, worker_action_data):
        super().save_eps(state, reward, action, done, next_state)
        self.worker_action_datas.append(worker_action_data)

    def save_replace_all(self, states, rewards, actions, dones, next_states, worker_action_datas):
        super().save_replace_all(states, rewards, actions, dones, next_states)
        self.worker_action_datas = worker_action_datas

    def clearMemory(self, idx = -100):
        super().clearMemory(idx)
        
        if idx == -100:
            del self.worker_action_datas[:]
        else:
            del self.worker_action_datas[idx]

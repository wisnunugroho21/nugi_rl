import tensorflow as tf
import numpy as np
from torch.utils.data import Dataset

class OnMemory:
    def __init__(self, datas = None):
        if datas is None :
            self.states         = np.array([], dtype = np.float32)
            self.actions        = np.array([], dtype = np.float32)
            self.rewards        = np.array([], dtype = np.float32)
            self.dones          = np.array([], dtype = np.float32)
            self.next_states    = np.array([], dtype = np.float32)
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

    def save_eps(self, state, action, reward, done, next_state):
        if self.__len__() == 0:
            self.states         = np.array([state], dtype = np.float32)
            self.actions        = np.array([action], dtype = np.float32)
            self.rewards        = np.array([[reward]], dtype = np.float32)
            self.dones          = np.array([[done]], dtype = np.float32)
            self.next_states    = np.array([next_state], dtype = np.float32)

        else:
            self.states         = np.append(self.states, [state], axis = 0)
            self.actions        = np.append(self.actions, [action], axis = 0)
            self.rewards        = np.append(self.rewards, [[reward]], axis = 0)
            self.dones          = np.append(self.dones, [[done]], axis = 0)
            self.next_states    = np.append(self.next_states, [next_state], axis = 0)

    def save_replace_all(self, states, actions, rewards, dones, next_states):
        self.states         = np.array(states)
        self.actions        = np.array(actions)
        self.rewards        = np.array(rewards)
        self.dones          = np.array(dones)
        self.next_states    = np.array(next_states)

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

    def clearMemory(self):
        self.states         = np.delete(self.states, np.s_[:])
        self.actions        = np.delete(self.actions, np.s_[:])
        self.rewards        = np.delete(self.rewards, np.s_[:])
        self.dones          = np.delete(self.dones, np.s_[:])
        self.next_states    = np.delete(self.next_states, np.s_[:])

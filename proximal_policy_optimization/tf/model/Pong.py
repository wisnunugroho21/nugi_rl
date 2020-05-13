import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

class Actor_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Actor_Model, self).__init__() 

        self.d1     = Dense(640, activation = "relu")
        self.d2     = Dense(640, activation = "relu")
        #self.d3     = Dense(640, activation = "relu")
        self.out    = Dense(action_dim, activation = "softmax")
        
    def call(self, states):
        x   = self.d1(states)
        x   = self.d2(x)
        #x   = self.d3(x)
        return self.out(x)

class Critic_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Critic_Model, self).__init__() 

        self.d1     = Dense(640, activation = "relu")
        self.d2     = Dense(640, activation = "relu")
        #self.d3     = Dense(640, activation = "relu")
        self.out    = Dense(1, activation = "linear")
        
    def call(self, states):
        x   = self.d1(states)
        x   = self.d2(x)
        #x   = self.d3(x)
        return self.out(x)

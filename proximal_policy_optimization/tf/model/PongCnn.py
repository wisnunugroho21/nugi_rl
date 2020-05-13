import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, ZeroPadding2D
from tensorflow.keras import Model
      
class Actor_Model(Model):
    def __init__(self, state_dim, action_dim):
      super(Actor_Model, self).__init__()   

      self.pad1   = ZeroPadding2D(padding = (2, 2))
      self.conv1  = Conv2D(32, kernel_size = 8, strides = (4, 4), padding = "valid", activation='relu')

      self.pad2   = ZeroPadding2D(padding = (1, 1))
      self.conv2  = Conv2D(32, kernel_size = 4, strides = (2, 2), padding = "valid", activation='relu')

      self.pad3   = ZeroPadding2D(padding = (1, 1))
      self.conv3  = Conv2D(32, kernel_size = 4, strides = (2, 2), padding = "valid", activation='relu')

      self.flat   = Flatten()

      self.d1     = Dense(400, activation = 'relu')
      self.out    = Dense(action_dim, activation = 'softmax')     
        
    def call(self, states):
      x = self.pad1(states)
      x = self.conv1(x)
      x = self.pad2(x)
      x = self.conv2(x)
      x = self.pad3(x)
      x = self.conv3(x)
      x = self.flat(x)
      x = self.d1(x)
      return self.out(x)

class Critic_Model(Model):
    def __init__(self, state_dim, action_dim):
      super(Critic_Model, self).__init__()

      self.pad1   = ZeroPadding2D(padding = (2, 2))
      self.conv1  = Conv2D(32, kernel_size = 8, strides = (4, 4), padding = "valid", activation='relu')

      self.pad2   = ZeroPadding2D(padding = (1, 1))
      self.conv2  = Conv2D(32, kernel_size = 4, strides = (2, 2), padding = "valid", activation='relu')

      self.pad3   = ZeroPadding2D(padding = (1, 1))
      self.conv3  = Conv2D(32, kernel_size = 4, strides = (2, 2), padding = "valid", activation='relu')

      self.flat   = Flatten()

      self.d1     = Dense(400, activation = 'relu')
      self.out    = Dense(1, activation = 'linear')
        
    def call(self, states):
      x = self.pad1(states)
      x = self.conv1(x)
      x = self.pad2(x)
      x = self.conv2(x)
      x = self.pad3(x)
      x = self.conv3(x)
      x = self.flat(x)
      x = self.d1(x)
      return self.out(x)
import gym
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def prepo(I):
    I = I[31:195, 8:152] # crop
    I = I[:,:, 0]
    I[I != 0] = 1
    #I[I == 142] = 0
    #print(I[100, 2])
    #I = I[35:195] # crop
    #I = I[:,:, 0]
    return I

env = gym.make('Breakout-v4')
for i_episode in range(20):
    done = False

    env.reset()
    obs = env.step(1)[0]

    for t in range(10000):
        env.render()        

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        if t > 100 or done:
            obs = prepo(obs)
            x = np.array(obs)
            print(x.shape)
            
            imgplot = plt.imshow(obs)
            plt.show()            

        if done:
            break
env.close()
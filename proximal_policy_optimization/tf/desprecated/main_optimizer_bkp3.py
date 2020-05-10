from utils.mongodb_utils import Observation, Weight
from ppo_agent.agent_continous_impala import Agent
from datetime import datetime
from mongoengine import *
import numpy as np
import time

def save_actor_weights(agent):
    Weight.objects.delete()
    actor_w = agent.get_weights()    
    actor_w = [np.array(w.tolist()) for w in actor_w]    

    for i1, w in enumerate(actor_w):
        for i2, a in enumerate(w):
            if not isinstance(a, np.ndarray):
                weightDb = Weight(weight = a.item(), dim1 = i1, dim2 = i2, dim3 = 0)
                weightDb.save()
                continue

            for i3, b in enumerate(a):
                weightDb = Weight(weight = b.item(), dim1 = i1, dim2 = i2, dim3 = i3)
                weightDb.save()

############## Hyperparameters ##############
training_mode       = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu             = True
reward_threshold    = None # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
render              = False # If you want to display the image. Turn this off if you run this in Google Collab
n_saved             = 100

n_plot_batch        = 100000 # How many episode you want to plot the result
n_episode           = 100000 # How many episode you want to run
n_update            = 1024 # How many episode before you update the Policy

policy_kl_range     = 0.03
policy_params       = 5
value_clip          = 1.0    
entropy_coef        = 0.0
vf_loss_coef        = 1.0
minibatch           = 32       
PPO_epochs          = 10
action_std          = 1.0
gamma               = 0.99
lam                 = 0.95
learning_rate       = 3e-4

env_name            = 'BipedalWalker-v3'
max_action          = 1.0
folder              = 'weights/humanoid_mujoco_w'

state_dim = 24 #8
action_dim = 4 #2

connect()

############################################# 
agent = Agent(state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
            minibatch, PPO_epochs, gamma, lam, learning_rate, action_std, folder, use_gpu)
print('Agent has been initialized')
#############################################

load_weights        = False
save_weights        = False
params_max          = 1.0
params_min          = 0.2
params_subtract     = 0.001
params_dynamic      = True
params              = params_max

#############################################
save_actor_weights(agent)
o = Observation.objects.delete()

while True:
    if Observation.objects.count() >= n_update:         
        print('Learning at: ', datetime.now().strftime("%H:%M:%S"), ' with length: ', Observation.objects.count())

        for obs in Observation.objects:
            agent.save_eps(obs.states, obs.actions, obs.rewards, obs.dones, obs.next_states, obs.logprobs, obs.next_next_states)
            obs.delete()
        
        #print('delete')
        print('length: ', Observation.objects.count(), ' length saved: ', len(agent.memory))

        agent.update_ppo()    

        if params_dynamic:
            params = params - params_subtract
            params = params if params > params_min else params_min  

        if save_weights:
            agent.save_weights() 
            print('weights saved')  

        save_actor_weights(agent)
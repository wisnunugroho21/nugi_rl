from flask import Flask, jsonify, request
from ppo_agent.agent_continuous import Agent
import redis
from utils.redis_utils import toRedis, fromRedis
import numpy as np

############## Hyperparameters ##############
training_mode       = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu             = True
reward_threshold    = None # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
render              = False # If you want to display the image. Turn this off if you run this in Google Collab
n_saved             = 100

n_plot_batch        = 100000 # How many episode you want to plot the result
n_episode           = 100000 # How many episode you want to run

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

env_name            = 'BipedalWalker-v2'
max_action          = 1.0
folder              = 'weights/humanoid_mujoco_w'

state_dim = 24 #8
action_dim = 4 #2

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
r = redis.Redis()
keys = ['states', 'rewards', 'actions', 'dones', 'next_states']
keys2 = ['actor_w1', 'actor_w2', 'actor_w3', 'actor_w4', 'actor_w5', 'actor_w6']

actor_w = agent.get_weights()    
actor_w = [w.tolist() for w in actor_w]

toRedis(r, np.array(actor_w[0]), 'actor_w1')
toRedis(r, np.array(actor_w[1]), 'actor_w2')
toRedis(r, np.array(actor_w[2]), 'actor_w3')
toRedis(r, np.array(actor_w[3]), 'actor_w4')
toRedis(r, np.array(actor_w[4]), 'actor_w5')
toRedis(r, np.array(actor_w[5]), 'actor_w6') 
r.set('is_optimizing', 0)
r.set('is_new_params', 0)

while True:
    if int(r.get('is_new_params').decode('utf-8')) == 1:
        r.set('is_optimizing', 1)
        print('Updating...')
        
        states       = fromRedis(r, 'states')
        rewards      = fromRedis(r, 'rewards')
        actions      = fromRedis(r, 'actions')
        dones        = fromRedis(r, 'dones')
        next_states  = fromRedis(r, 'next_states')
        agent.save_replace_all_eps(states.tolist(), rewards.tolist(), actions.tolist(), dones.tolist(), next_states.tolist())     
        r.set('is_new_params', 0)   

        agent.update_ppo()    

        if params_dynamic:
            params = params - params_subtract
            params = params if params > params_min else params_min  

        if save_weights:
            agent.save_weights() 
            print('weights saved')  

        actor_w = agent.get_weights() 
        actor_w = [w.tolist() for w in actor_w]

        toRedis(r, np.array(actor_w[0]), 'actor_w1')
        toRedis(r, np.array(actor_w[1]), 'actor_w2')
        toRedis(r, np.array(actor_w[2]), 'actor_w3')
        toRedis(r, np.array(actor_w[3]), 'actor_w4')
        toRedis(r, np.array(actor_w[4]), 'actor_w5')
        toRedis(r, np.array(actor_w[5]), 'actor_w6')       

        r.set('is_optimizing', 0)  
        print('Finish Updating...')
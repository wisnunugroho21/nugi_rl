from flask import Flask, jsonify, request
from ppo_agent.agent_continous_impala import AgentContinous
from model.BasicTanhNN import Actor_Model, Critic_Model
from memory.on_policy_impala_memory import OnMemoryImpala
from utils.mongodb_utils import Observation, Weight

import socketio
import requests
import numpy as np
from mongoengine import *

############## Hyperparameters ##############
training_mode       = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu             = True
reward_threshold    = None # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
load_weights        = False
save_weights        = False

policy_kl_range     = 0.05
policy_params       = 5.0
value_clip          = 5.0    
entropy_coef        = 0.0
vf_loss_coef        = 1.0
minibatch           = 32       
PPO_epochs          = 10
action_std          = 1.0
gamma               = 0.99
lam                 = 0.95
learning_rate       = 3e-4

env_name            = 'BipedalWalker-v3'
folder              = 'weights/bipedal_multi_agent'

state_dim = 24 #8
action_dim = 4 #2

#############################################

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

def set_actor_weights(agent):
    actor_w = []
    actor_w.append(np.ones((64, 24)))
    actor_w.append(np.ones((64,)))
    actor_w.append(np.ones((64, 64)))
    actor_w.append(np.ones((64,)))
    actor_w.append(np.ones((64, 64)))
    actor_w.append(np.ones((64,)))
    actor_w.append(np.ones((4, 64)))
    actor_w.append(np.ones((4,)))

    if Weight.objects.count() > 0:
        for w in Weight.objects:
            aw = actor_w[w.dim1]

            if len(aw.shape) == 1:
                aw[w.dim2,] = w.weight
            elif len(aw.shape) == 2:
                aw[w.dim2, w.dim3] = w.weight

        agent.set_weights(actor_w)

    return agent

#############################################
app                         = Flask(__name__)
app.config['SECRET_KEY']    = 'vnkdjnfjknfl1232#'
sio                         = socketio.Client()
connect()

agent = AgentContinous(Actor_Model, Critic_Model, state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
            minibatch, PPO_epochs, gamma, lam, learning_rate, action_std, folder, use_gpu)
save_actor_weights(agent)

print('Agent has been initialized')
#############################################

@app.route('/act', methods=['POST'])
def act():
    global agent
    agent = set_actor_weights(agent)    
                
    data    = request.get_json()
    state   = data['state']
    action, worker_action_datas = agent.act(state)

    data = {
        'action'                : action.tolist(),
        'worker_action_datas'   : worker_action_datas.tolist()
    }

    return jsonify(data)        

@sio.event
def update_model():
    global agent
    agent = set_actor_weights(agent)

    r = requests.get(url = 'http://localhost:5000/trajectory')
    data = r.json()

    states              = data['states']
    actions             = data['actions']
    rewards             = data['rewards']    
    dones               = data['dones']
    next_states         = data['next_states']
    worker_action_datas = data['worker_action_datas']
    
    agent.memory.save_replace_all(states, actions, rewards, dones, next_states, worker_action_datas)
    agent.update_ppo()
    save_actor_weights(agent)

    if save_weights:
        agent.save_weights()
        print('weights saved')        
        

@app.route('/test')
def test():
    return 'test'

@sio.event
def connect():
    print('connect')

@sio.event
def disconnect():
    print('disconnect')

@sio.event
def reconnect():
    print('reconnect')

sio.connect('http://localhost:5000')
print('my sid is', sio.sid)

app.run(host = 'localhost', port = 8010, threaded = True)
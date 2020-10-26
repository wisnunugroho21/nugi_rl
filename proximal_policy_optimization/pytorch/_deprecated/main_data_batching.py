from flask import Flask, jsonify, request
from flask_socketio import SocketIO, send, emit
from datetime import datetime
import numpy as np
import requests
from mongoengine import *

from memory.on_policy_impala_memory import OnMemoryImpala
from utils.mongodb_utils import Observation, Weight

############## Hyperparameters ##############
render      = False # If you want to display the image. Turn this off if you run this in Google Collab
n_update    = 1024 # How many episode before you update the Policy
state_dim   = 24 #8
action_dim  = 4 #2
#############################################
app                         = Flask(__name__)
app.config['SECRET_KEY']    = 'vnkdjnfjknfl1232#'
socketio                    = SocketIO(app)
connect()
Observation.objects.delete()

print('Agent has been initialized')
############################################# 

@app.route('/trajectory', methods=['POST'])
def save_trajectory():
    data = request.get_json()    

    states              = data['states']
    actions             = data['actions']
    rewards             = data['rewards']
    dones               = data['dones']
    next_states         = data['next_states']
    worker_action_datas = data['worker_action_datas']

    for s, a, r, d, ns, wad in zip(states, actions, rewards, dones, next_states, worker_action_datas):            
        obs = Observation(states = s, actions = a, rewards = r, dones = d, next_states = ns, worker_action_datas = wad)
        obs.save()

    if Observation.objects.count() >= n_update:
        socketio.emit('update_model')

    data = {
        'success': True
    }

    return jsonify(data)

@app.route('/trajectory', methods=['GET'])
def send_trajectory():
    memory = OnMemoryImpala()

    for obs in Observation.objects:
        memory.save_eps(obs.states, obs.actions, obs.rewards, obs.dones, obs.next_states, obs.worker_action_datas)
        obs.delete()
    
    states, actions, rewards, dones, next_states, worker_action_datas = memory.get_all_items()
    data = {
        'states'                : states,
        'actions'               : actions,
        'rewards'               : rewards,        
        'dones'                 : dones,
        'next_states'           : next_states,
        'worker_action_datas'   : worker_action_datas
    }

    memory.clear_memory()
    return jsonify(data)

@app.route('/test')
def test():
    return 'test'

if __name__ == '__main__':
    print('Run..')
    socketio.run(app)
#app.run(host = '0.0.0.0', port = 8010, debug = True, threaded = True)
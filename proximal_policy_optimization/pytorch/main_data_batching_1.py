from flask import Flask, jsonify, request
from flask_socketio import SocketIO, send, emit
from datetime import datetime
import numpy as np
import requests

from memory.on_policy_impala_memory import OnMemory

############## Hyperparameters ##############
render      = False # If you want to display the image. Turn this off if you run this in Google Collab
n_update    = 128 # How many episode before you update the Policy
state_dim   = 24 #8
action_dim  = 4 #2
############################################# 
memory = OnMemory()
print('Agent has been initialized')
#############################################
app                         = Flask(__name__)
app.config['SECRET_KEY']    = 'vnkdjnfjknfl1232#'
socketio                    = SocketIO(app)

@app.route('/act', methods=['POST'])
def act():
    global memory

    data = request.get_json()
    data = {
        'state': data['state']
    }

    r = requests.post(url = 'http://localhost:8010/act', json = data)
    data = r.json()

    return data

@app.route('/trajectory', methods=['POST'])
def save_trajectory():
    global memory

    data = request.get_json()

    states              = data['states']
    actions             = data['actions']
    rewards             = data['rewards']
    dones               = data['dones']
    next_states         = data['next_states']
    logprobs            = data['logprobs']
    next_next_states    = data['next_next_states']

    if isinstance(dones, list):
        for s, a, r, d, ns, l, nns in zip(states, actions, rewards, dones, next_states, logprobs, next_next_states):
            memory.save_eps(s, a, r, d, ns, l, nns)
    else:
        memory.save_eps(states, actions, rewards, dones, next_states, logprobs)

    if len(memory) >= n_update:
        socketio.emit('update')

    data = {
        'success': True
    }

    return jsonify(data)

@app.route('/trajectory', methods=['GET'])
def send_trajectory():
    global memory

    #memory.convert_next_states_to_next_next_states()
    states             = []
    actions            = []
    rewards            = []
    dones              = []
    next_states        = []
    logprobs           = []
    next_next_states   = []

    for i in range(len(memory)):
        state, action, reward, done, next_state, logprob, next_next_state = memory.pop(0)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        next_states.append(next_state)
        logprobs.append(logprob)
        next_next_states.append(next_next_state)
    
    data = {
        'states'            : states,
        'actions'           : actions,
        'rewards'           : rewards,        
        'dones'             : dones,
        'next_states'       : next_states,
        'logprobs'          : logprobs,
        'next_next_states'  : next_next_states
    }

    memory.clearMemory()
    return jsonify(data)

def update(message):
    global memory

    print('start updating at ', datetime.now().strftime("%H:%M:%S"))

    states, actions, rewards, dones, next_states, logprobs, next_next_states = memory.get_all_items()
    data = {
        'states'            : states,
        'actions'           : actions,
        'rewards'           : rewards,        
        'dones'             : dones,
        'next_states'       : next_states,
        'logprobs'          : logprobs,
        'next_next_states'  : next_next_states
    }

    r = requests.post(url = 'http://localhost:8010/update', json = data)
    data = r.json()

    memory.clearMemory()
    print('finish updating at ', datetime.now().strftime("%H:%M:%S"))

    return data


@app.route('/test')
def test():
    return 'test'

if __name__ == '__main__':
    print('Run..')
    socketio.run(app)
#app.run(host = '0.0.0.0', port = 8010, debug = True, threaded = True)
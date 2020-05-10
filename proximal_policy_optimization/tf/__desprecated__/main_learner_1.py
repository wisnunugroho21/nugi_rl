from flask import Flask, jsonify, request
from ppo_agent.agent_continous_impala import AgentContinous
from memory.on_policy_impala_memory import OnMemoryImpala
import socketio
import requests

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

agent = AgentContinous(state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
            minibatch, PPO_epochs, gamma, lam, learning_rate, action_std, folder, use_gpu)
print('Agent has been initialized')

#############################################
app                         = Flask(__name__)
app.config['SECRET_KEY']    = 'vnkdjnfjknfl1232#'
sio                         = socketio.Client()

@app.route('/update', methods=['POST'])
def update_model_via_api():
    global agent
    global save_weights

    data = request.get_json()

    states              = data['states']
    actions             = data['actions']
    rewards             = data['rewards']    
    dones               = data['dones']
    next_states         = data['next_states']
    worker_action_datas = data['worker_action_datas']
    
    agent.memory.save_replace_all_eps(states, actions, rewards, dones, next_states, worker_action_datas)
    agent.update_ppo()    

    if save_weights:
        agent.save_weights()
        print('weights saved')  

    data = {
        'success': True
    }

    return jsonify(data)

@app.route('/act', methods=['POST'])
def act():
    global agent

    data            = request.get_json()
    state           = data['state']
    action, logprob = agent.act(state)

    data = {
        'action': action.tolist(),
        'logprob': logprob.tolist()
    }

    return jsonify(data)

@sio.event
def update():
    global agent
    global save_weights
    
    r = requests.get(url = 'http://localhost:5000/trajectory')
    data = r.json()

    states              = data['states']
    actions             = data['actions']
    rewards             = data['rewards']    
    dones               = data['dones']
    next_states         = data['next_states']
    worker_action_datas = data['worker_action_datas']
    
    agent.memory.save_replace_all_eps(states, actions, rewards, dones, next_states, worker_action_datas)
    agent.update_ppo()    

    if save_weights:
        agent.save_weights()
        print('weights saved')  

    data = {
        'success': True
    }

    return jsonify(data)

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
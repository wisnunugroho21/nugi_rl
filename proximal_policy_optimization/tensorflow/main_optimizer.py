from flask import Flask, jsonify, request
from ppo_agent.agent import Agent
import tensorflow as tf

memory_gpu = 2048

#############################################

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = memory_gpu)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

############## Hyperparameters ##############
load_weights = False # If you want to load the agent, set this to True
save_weights = False # If you want to save the agent, set this to True
training_mode = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it

policy_kl_range = 0.03
policy_params = 5
value_clip = 1.0    
entropy_coef = 0.0
vf_loss_coef = 1.0
minibatch = 32       
PPO_epochs = 10
action_std = 1.0

gamma = 0.99
lam = 0.95
learning_rate = 3e-4

params_max = 1.0
params_min = 0.2
params_subtract = 0.00001
params_dynamic = True

state_dim = 24 #8
action_dim = 4 #2
############################################# 

params = params_max    
agent = Agent(action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                minibatch, PPO_epochs, gamma, lam, learning_rate, action_std)
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

app                 = Flask(__name__)

@app.route('/update_policy')
def update_policy():
    global agent
    global params
    global save_weights

    data = request.get_json()

    states       = data['states']
    rewards      = data['rewards']
    actions      = data['actions']
    dones        = data['dones']
    next_states  = data['next_states']

    agent.save_replace_all_eps(states, rewards, actions, dones, next_states)
    agent.update_ppo()    

    if params_dynamic:
        params = params - params_subtract
        params = params if params > params_min else params_min  

    if save_weights:
        agent.save_weights() 
        print('weights saved')  

    actor_w = agent.get_weights()
    actor_w = [w.tolist() for w in actor_w]

    data = {
        'actor': actor_w
    }

    return jsonify(data)

@app.route('/test')
def test():
    return 'test'

#app.run(host = '0.0.0.0', port = 8010, debug = True, threaded = True)
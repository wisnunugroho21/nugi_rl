from flask import Flask, jsonify, request
from ppo_agent.agent_continous_impala import Agent
from memory.on_policy_impala_memory import OnMemory

############## Hyperparameters ##############
training_mode       = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu             = True
reward_threshold    = None # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

render              = False # If you want to display the image. Turn this off if you run this in Google Collab
n_saved             = 100

n_plot_batch        = 100000 # How many episode you want to plot the result
n_episode           = 100000 # How many episode you want to run

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
max_action          = 1.0
folder              = 'weights/bipedal_multi_agent'

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

app                 = Flask(__name__)

@app.route('/update_policy')
def update_policy():
    global agent
    global params
    global save_weights

    data = request.get_json()

    states          = data['states']
    rewards         = data['rewards']
    actions         = data['actions']
    dones           = data['dones']
    next_states     = data['next_states']
    worker_logprobs = data['worker_logprobs']

    memory = OnMemory((states, rewards, actions, dones, next_states, worker_logprobs))
    agent.update_ppo(memory)    

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

@app.route('/get_weights')
def get_weights():
    global agent

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
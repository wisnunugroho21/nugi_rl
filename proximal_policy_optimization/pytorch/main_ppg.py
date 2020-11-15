import gym

from eps_runner.ppg.standard import run_discrete_episode, run_continous_episode
from executor.ppg.standard import run_discrete, run_continous

from agent.ppg.agent_standard import AgentDiscrete, AgentContinous
from model.PPGTanhNN import Policy_Model, Value_Model

############## Hyperparameters ##############

load_weights        = False # If you want to load the agent, set this to True
save_weights        = False # If you want to save the agent, set this to True
training_mode       = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu             = True
reward_threshold    = 300 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

render              = True # If you want to display the image. Turn this off if you run this in Google Collab
n_saved             = 10

n_plot_batch        = 100000 # How many episode you want to plot the result
n_episode           = 100000 # How many episode you want to run

n_update            = 1024 # How many episode before you update the Policy
n_aux_update        = 5
policy_kl_range     = 0.03
policy_params       = 5
value_clip          = 5.0
entropy_coef        = 0.0
vf_loss_coef        = 1.0
batch_size          = 32 
PPO_epochs          = 10
Aux_epochs          = 10
action_std          = 1.0
gamma               = 0.99
lam                 = 0.95
learning_rate       = 5e-4

params_max          = 1.0
params_min          = 0.25
params_subtract     = 0.001
params_dynamic      = False

env_name            = 'BipedalWalker-v3'
max_action          = 1.0
folder              = 'weights/pong_lstm1'

############################################# 

env = gym.make(env_name)
#env = SumoEnv()
#env = ContinuousCartPoleEnv()

if type(env.observation_space) is gym.spaces.Box:
    if len(env.observation_space.shape) > 1:
        state_dim = 1
        for i in range(len(env.observation_space.shape)):
            state_dim *= env.observation_space.shape[i]
    
    else:
        state_dim = env.observation_space.shape[0]
            
else:
    state_dim = env.observation_space.n

#print(env.unwrapped.get_action_meanings())
#state_dim = 80 * 80
print('state_dim: ', state_dim)

if type(env.action_space) is gym.spaces.Box:
    action_dim = env.action_space.shape[0]    

    print('action_dim: ', action_dim)

    agent = AgentContinous(Policy_Model, Value_Model, state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
            batch_size, PPO_epochs, Aux_epochs, gamma, lam, learning_rate, action_std, folder, use_gpu)

    if load_weights:
        agent.load_weights()
        print('Weight Loaded')

    run_continous(agent, env, n_episode, run_continous_episode, reward_threshold, save_weights, n_plot_batch, render, training_mode, n_update, n_aux_update, n_saved,
       params_max, params_min, params_subtract, params_dynamic, max_action)

else:
    action_dim = env.action_space.n
    #action_dim = 3

    print('action_dim: ', action_dim)

    agent = AgentDiscrete(Policy_Model, Value_Model, state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
           batch_size, PPO_epochs, Aux_epochs, gamma, lam, learning_rate, folder, use_gpu)

    if load_weights:
        agent.load_weights()
        print('Weight Loaded')

    run_discrete(agent, env, n_episode, run_discrete_episode, reward_threshold, save_weights, n_plot_batch, render, training_mode, n_update, n_aux_update, n_saved,
        params_max, params_min, params_subtract, params_dynamic)
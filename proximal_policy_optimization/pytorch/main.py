import gym

from eps_runner.ppg.pong_eps import PongRunner
from executor.ppg.standard import StandardExecutor

from agent.ppg.agent_standard import AgentDiscrete, AgentContinous
from model.ppg.Pong import Policy_Model, Value_Model

from run import run

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = True # If you want to save the agent, set this to True
training_mode           = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu                 = True
reward_threshold        = 20 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

render                  = True # If you want to display the image. Turn this off if you run this in Google Collab
n_saved                 = 10

n_plot_batch            = 100000 # How many episode you want to plot the result
n_episode               = 100000 # How many episode you want to run
n_update                = 256 # How many episode before you update the Policy
n_aux_update            = 10

policy_kl_range         = 0.0008
policy_params           = 20
value_clip              = 2.0
entropy_coef            = 0.01
vf_loss_coef            = 1.0
batch_size              = 32
PPO_epochs              = 4
Aux_epochs              = 4
action_std              = 1.0
gamma                   = 0.99
lam                     = 0.95
learning_rate           = 2.5e-4

params_max              = 1.0
params_min              = 0.25
params_subtract         = 0.001
params_dynamic          = False

env_name                = 'Pong-v4'
max_action              = 1.0
folder                  = 'weights/pong'

use_ppg                 = True

Policy_or_Actor_Model   = Policy_Model
Value_or_Critic_Model   = Value_Model
Runner                  = PongRunner
Executor                = StandardExecutor

state_dim               = 80 * 80
action_dim              = 3

env                     = gym.make(env_name)

#############################################  

run(Runner, Executor, AgentDiscrete, AgentContinous, Policy_or_Actor_Model, Value_or_Critic_Model, env, state_dim, action_dim,
    load_weights, save_weights, training_mode, use_gpu, reward_threshold, render, n_saved, n_plot_batch, n_episode, n_update, n_aux_update,
    policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef, batch_size, PPO_epochs, Aux_epochs, action_std, gamma, lam, learning_rate,
    params_max, params_min, params_subtract, params_dynamic, max_action, folder, use_ppg)
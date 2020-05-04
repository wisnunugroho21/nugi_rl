from eps_runner.cartpole_continous_eps import run
from ppo_agent.agent_continuous import Agent
from env.cartpole_continous import ContinuousCartPoleEnv

import gym
from gym.envs.registration import register

import tensorflow as tf

############## Hyperparameters ##############

load_weights = False # If you want to load the agent, set this to True
save_weights = False # If you want to save the agent, set this to True
training_mode = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
reward_threshold = None # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

render = True # If you want to display the image. Turn this off if you run this in Google Collab
n_update = 32 # How many episode before you update the Policy
n_plot_batch = 100000 # How many episode you want to plot the result
n_episode = 100000 # How many episode you want to run
n_saved = 10

policy_kl_range = 0.03
policy_params = 5
value_clip = 2.0    
entropy_coef = 0.0
vf_loss_coef = 1.0
minibatch = 2       
PPO_epochs = 4
action_std = 1.0

gamma = 0.99
lam = 0.95
learning_rate = 3e-4

params_max = 1.0
params_min = 0.2
params_subtract = 0.0000001
params_dynamic = True

env_name = 'CartPole-v0'
folder = 'weights/traffic_lights_sumo'
max_action = 1.0

############################################# 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

############################################# 
#env = gym.make(env_name)
#env = SumoEnv()
env = ContinuousCartPoleEnv()

#state_dim = env.observation_dim
#action_dim = env.action_dim

state_dim = env.observation_space.shape[0]
#state_dim = env.observation_space.n
action_dim = env.action_space.shape[0]
#action_dim = env.action_space.n

agent = Agent(action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
            minibatch, PPO_epochs, gamma, lam, learning_rate, action_std, folder)

#state_dim = 80 * 80
#action_dim = 2
#print(env.unwrapped.get_action_meanings())

#agent = Agent(action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
#           minibatch, PPO_epochs, gamma, lam, learning_rate, folder)            

if load_weights:
    agent.load_weights()
    print('Weight Loaded')

run(agent, env, n_episode, reward_threshold, save_weights, n_plot_batch, render, training_mode, n_update, n_saved,
       params_max, params_min, params_subtract, params_dynamic, max_action)

#run(agent, env, n_episode, reward_threshold, save_weights, n_plot_batch, render, training_mode, n_update, n_saved,
#        params_max, params_min, params_subtract, params_dynamic)
from eps_runner.worker_continous_eps import run
import gym

############## Hyperparameters ##############

training_mode       = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
reward_threshold    = None # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
render              = True # If you want to display the image. Turn this off if you run this in Google Collab
n_saved             = 100

n_plot_batch        = 100000 # How many episode you want to plot the result
n_episode           = 100000 # How many episode you want to run

env_name            = 'BipedalWalker-v3'
max_action          = 1.0
folder              = 'weights/bipedal_multi_agent'

#############################################

env = gym.make(env_name)
run(env, n_episode, reward_threshold, n_plot_batch, render, training_mode, max_action)

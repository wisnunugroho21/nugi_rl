import gym
import random
import numpy as np
from sympy import im
import torch
import os

from torch.utils.tensorboard import SummaryWriter
from torch.optim.adamw import AdamW

from nugi_rl.train.runner.iteration.pong import PongRunner
from nugi_rl.train.executor.standard import Executor
from nugi_rl.agent.ppo import AgentPPO
from nugi_rl.distribution.discrete.basic import BasicDiscrete
from nugi_rl.environment.wrapper.gym_wrapper import GymWrapper
from nugi_rl.loss.ppo.truly_ppo import TrulyPpo
from nugi_rl.loss.value import ValueLoss
from nugi_rl.loss.entropy import EntropyLoss
from nugi_rl.policy_function.advantage_function.gae import GeneralizedAdvantageEstimation
from nugi_rl.model.ppo.SoftmaxNN import Policy_Model, Value_Model
from nugi_rl.memory.policy.standard import PolicyMemory
from nugi_rl.helpers.plotter.weight_bias import WeightBiasPlotter

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = True # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
render                  = True # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 1000 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 100000000 # How many episode you want to run
n_update                = 1024 # How many episode before you update the Policy 
n_saved                 = 1

policy_kl_range         = 0.03
policy_params           = 5
value_clip              = None
entropy_coef            = 0.1
vf_loss_coef            = 1.0
batch_size              = 32
epochs                  = 5
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 3e-4

device_name             = 'cuda'
env_name                = 'Pong-v4'
folder                  = 'weights'

state_dim               = 80 * 80
action_dim              = 3
max_action              = 1

config = { load_weights, save_weights, is_training_mode, render, reward_threshold, n_plot_batch, n_iteration,
    n_update, n_saved, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef, batch_size,
    epochs, action_std, gamma, learning_rate, device_name, env_name
}

#####################################################################################################################################################

device              = torch.device(device_name)

environment         = GymWrapper(gym.make(env_name), device)
distribution        = BasicDiscrete()
advantage_function  = GeneralizedAdvantageEstimation(gamma)
plotter             = WeightBiasPlotter(config, 'Pong_v2', entity = "wisnunugroho21")

memory          = PolicyMemory()
ppo_loss        = TrulyPpo(distribution, policy_kl_range, policy_params)
value_loss      = ValueLoss(vf_loss_coef, value_clip)
entropy_loss    = EntropyLoss(distribution, entropy_coef)

policy          = Policy_Model(state_dim, action_dim).float().to(device)
value           = Value_Model(state_dim).float().to(device)
optimizer       = AdamW(list(policy.parameters()) + list(value.parameters()), lr = learning_rate)

agent   = AgentPPO(policy, value, advantage_function, distribution, ppo_loss, value_loss, entropy_loss, memory, optimizer, 
    epochs, is_training_mode, batch_size, folder, device)

runner      = PongRunner(agent, environment, is_training_mode, render, n_update, plotter, n_plot_batch)
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()
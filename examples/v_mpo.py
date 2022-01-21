import gym
import random
import numpy as np
import torch
import os
import robosuite as suite
from robosuite.wrappers import GymWrapper as RobosuiteWrapper

from torch.utils.tensorboard import SummaryWriter
from torch.optim.adamw import AdamW

from nugi_rl.train.runner.iteration.standard import IterRunner
from nugi_rl.train.executor.standard import Executor
from nugi_rl.agent.v_mpo import AgentVMPO
from nugi_rl.distribution.continous.multivariate import MultivariateContinous
from nugi_rl.environment.wrapper.gym_wrapper import GymWrapper
from nugi_rl.loss.v_mpo.phi_loss import PhiLoss
from nugi_rl.loss.v_mpo.temperature_loss import TemperatureLoss
from nugi_rl.loss.v_mpo.alpha.continous_alpha_loss import ContinuousAlphaLoss
from nugi_rl.loss.value import ValueLoss
from nugi_rl.loss.entropy import EntropyLoss
from nugi_rl.policy_function.advantage_function.gae import GeneralizedAdvantageEstimation
from nugi_rl.model.v_mpo.TanhNN import Policy_Model, Value_Model
from nugi_rl.memory.policy.standard import PolicyMemory

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = False # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
render                  = True # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 1000 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 100000000 # How many episode you want to run
n_update                = 2048 # How many episode before you update the Policy
n_saved                 = 1

coef_alpha_mean_upper   = 0.01
coef_alpha_mean_below   = 0.005
coef_alpha_cov_upper    = 5e-5
coef_alpha_cov_below    = 5e-6

# coef_alpha_upper        = 0.01
# coef_alpha_below        = 0.005

coef_temp               = 0.01
batch_size              = 64
policy_epochs           = 5
value_clip              = None
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 1e-4
entropy_coef            = 0.1
vf_loss_coef            = 1.0

device                  = torch.device('cuda')
folder                  = 'weights/v_mpo_lift'

env = gym.make('BipedalWalker-v3')

# env                     = RobosuiteWrapper(
#     suite.make(
#         "Lift",
#         robots="Sawyer",                # use Sawyer robot
#         use_camera_obs=False,           # do not use pixel observations
#         has_offscreen_renderer=False,   # not needed since not using pixel obs
#         has_renderer=True,              # make sure we can render to the screen
#         reward_shaping=True,            # use dense rewards
#         control_freq=20,                # control should happen fast enough so that simulation looks smooth
#     )
# )

state_dim               = None
action_dim              = None
max_action              = 1

#####################################################################################################################################################

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)
os.environ['PYTHONHASHSEED'] = str(20)

environment         = GymWrapper(env, device)

if state_dim is None:
    state_dim = environment.get_obs_dim()
print('state_dim: ', state_dim)

if environment.is_discrete():
    print('discrete')
else:
    print('continous')

if action_dim is None:
    action_dim = environment.get_action_dim()
print('action_dim: ', action_dim)

coef_alpha_mean_upper   = torch.Tensor([coef_alpha_mean_upper]).to(device)
coef_alpha_mean_below   = torch.Tensor([coef_alpha_mean_below]).to(device)
coef_alpha_cov_upper    = torch.Tensor([coef_alpha_cov_upper]).to(device)
coef_alpha_cov_below    = torch.Tensor([coef_alpha_cov_below]).to(device)

# coef_alpha_upper    = torch.Tensor([coef_alpha_upper]).to(device)
# coef_alpha_below    = torch.Tensor([coef_alpha_below]).to(device)

distribution        = MultivariateContinous()
advantage_function  = GeneralizedAdvantageEstimation(gamma)
policy_memory       = PolicyMemory(device)

alpha_loss          = ContinuousAlphaLoss(distribution, coef_alpha_mean_upper, coef_alpha_mean_below, coef_alpha_cov_upper, coef_alpha_cov_below)
phi_loss            = PhiLoss(distribution)
temperature_loss    = TemperatureLoss(coef_temp)
value_loss          = ValueLoss(vf_loss_coef, value_clip)
entropy_loss        = EntropyLoss(distribution, entropy_coef)

policy              = Policy_Model(state_dim, action_dim).to(device)
value               = Value_Model(state_dim).to(device)
policy_optimizer    = AdamW(policy.parameters(), lr = learning_rate)
value_optimizer     = AdamW(value.parameters(), lr = learning_rate)

agent   = AgentVMPO(policy, value, advantage_function, distribution, alpha_loss, phi_loss, entropy_loss, temperature_loss, value_loss,
            policy_memory, policy_optimizer, value_optimizer, policy_epochs, is_training_mode, batch_size, folder, device)

runner      = IterRunner(agent, environment, is_training_mode, render, n_update, SummaryWriter(), n_plot_batch)
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()
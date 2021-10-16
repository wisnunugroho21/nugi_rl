import gym
import random
import numpy as np
import torch
import os
import pybullet_envs
# from pybullet_envs.deep_mimic.gym_env.deep_mimic_env import HumanoidDeepMimicWalkBulletEnv

from torch.utils.tensorboard import SummaryWriter
from torch.optim.adamw import AdamW

from eps_runner.iteration.tensor import TensorIterRunner
from train_executor.executor import Executor
from agent.tensor.v_mpo import AgentTensorVMPO
from distribution.multivariate_continous import MultivariateContinous
from environment.wrapper.brax_wrapper import BraxWrapper
from loss.v_mpo.phi_loss import PhiLoss
from loss.v_mpo.temperature_loss import TemperatureLoss
from loss.v_mpo.continous_alpha_loss import AlphaLoss
from loss.v_mpo.value_loss import ValueLoss
from loss.v_mpo.entropy_loss import EntropyLoss
from policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from model.v_mpo.TanhNN import Policy_Model, Value_Model
from memory.policy.tensor import TensorPolicyMemory

############## Hyperparameters ##############

load_weights            = True # If you want to load the agent, set this to True
save_weights            = True # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
render                  = False # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 1000 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 100000000 # How many episode you want to run
n_update                = 1024 # How many episode before you update the Policy
n_saved                 = 1

coef_alpha_mean_upper   = 0.01
coef_alpha_mean_below   = 0.005
coef_alpha_cov_upper    = 5e-5
coef_alpha_cov_below    = 5e-6

coef_temp               = 0.01
batch_size              = 64
policy_epochs           = 10
value_clip              = None
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 2e-4
entropy_coef            = 0.1

device                  = torch.device('cuda')
folder                  = 'weights/v_mpo_humanoid'

BraxWrapper.register_brax_gym(gym)
env                     = gym.make('brax_humanoid-v0') # gym.make('BipedalWalker-v3') #gym.make("HumanoidBulletEnv-v0") # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]
# env.render()

state_dim               = None
action_dim              = None
max_action              = 1

#####################################################################################################################################################

random.seed(30)
np.random.seed(30)
torch.manual_seed(30)
os.environ['PYTHONHASHSEED'] = str(30)

environment = BraxWrapper(env)

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

distribution        = MultivariateContinous()
advantage_function  = GeneralizedAdvantageEstimation(gamma)
policy_memory       = TensorPolicyMemory(device)

alpha_loss          = AlphaLoss(distribution, coef_alpha_mean_upper, coef_alpha_mean_below, coef_alpha_cov_upper, coef_alpha_cov_below)
phi_loss            = PhiLoss(distribution, advantage_function)
temperature_loss    = TemperatureLoss(advantage_function, coef_temp, device)
value_loss          = ValueLoss(advantage_function, value_clip)
entropy_loss        = EntropyLoss(distribution, entropy_coef)

policy              = Policy_Model(state_dim, action_dim).float().to(device)
value               = Value_Model(state_dim).float().to(device)
policy_optimizer    = AdamW(policy.parameters(), lr = learning_rate)
value_optimizer     = AdamW(value.parameters(), lr = learning_rate)

agent   = AgentTensorVMPO(policy, value, distribution, alpha_loss, phi_loss, entropy_loss, temperature_loss, value_loss,
            policy_memory, policy_optimizer, value_optimizer, policy_epochs, is_training_mode, batch_size, folder, device)

runner      = TensorIterRunner(agent, environment, is_training_mode, render, n_update, environment.is_discrete, max_action, SummaryWriter(), n_plot_batch)
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()
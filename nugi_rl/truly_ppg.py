import gym
import random
import numpy as np
import torch
import os
from pybullet_envs.deep_mimic.gym_env.deep_mimic_env import HumanoidDeepMimicWalkBulletEnv

from torch.utils.tensorboard import SummaryWriter
from torch.optim.adamw import AdamW

from eps_runner.iteration.iter_runner import IterRunner
from train_executor.executor import Executor
from agent.standard.ppg import AgentPPG
from distribution.basic_continous import BasicContinous
from environment.wrapper.gym_wrapper import GymWrapper
from loss.other.aux_ppg import AuxPPG
from loss.trpo_ppo.truly_ppo import TrulyPPO
from policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from model.ppg.TanhStdNN import Policy_Model, Value_Model
from memory.policy.standard import PolicyMemory
from memory.aux_ppg.standard import AuxPpgMemory

############## Hyperparameters ##############

load_weights            = True # If you want to load the agent, set this to True
save_weights            = True # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
render                  = True # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 1000 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 100000000 # How many episode you want to run
n_update                = 2048 # How many episode before you update the Policy
n_aux_update            = 20
n_saved                 = n_aux_update

policy_kl_range         = 0.05
policy_params           = 5
value_clip              = None
entropy_coef            = 0.2
vf_loss_coef            = 1.0
batch_size              = 32
ppo_epochs              = 10
aux_ppg_epochs          = 10
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 3e-4

device                  = torch.device('cuda:0')
folder                  = 'weights/truly_ppg_humanoid_mimic_walk'

env                     = HumanoidDeepMimicWalkBulletEnv(renders = render) # gym.make("KukaBulletEnv-v0") # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]
env.render(mode = "human")

state_dim               = None
action_dim              = None
max_action              = 1

#####################################################################################################################################################

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)
os.environ['PYTHONHASHSEED'] = str(20)

environment         = GymWrapper(env)

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

distribution        = BasicContinous()
advantage_function  = GeneralizedAdvantageEstimation(gamma)

aux_ppg_memory      = AuxPpgMemory()
ppo_memory          = PolicyMemory()

aux_ppg_loss        = AuxPPG(distribution)
ppo_loss            = TrulyPPO(distribution, advantage_function, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef)

policy              = Policy_Model(state_dim, action_dim).float().to(device)
value               = Value_Model(state_dim).float().to(device)
ppo_optimizer       = AdamW(list(policy.parameters()) + list(value.parameters()), lr = learning_rate)        
aux_ppg_optimizer   = AdamW(list(policy.parameters()), lr = learning_rate)

agent   = AgentPPG(policy, value, distribution, ppo_loss, aux_ppg_loss, ppo_memory, aux_ppg_memory, 
                ppo_optimizer, aux_ppg_optimizer, ppo_epochs, aux_ppg_epochs, n_aux_update, is_training_mode, 
                batch_size,  folder, device)

runner      = IterRunner(agent, environment, is_training_mode, render, n_update, environment.is_discrete(), max_action, SummaryWriter(), n_plot_batch) # [Runner.remote(i_env, render, training_mode, n_update, Wrapper.is_discrete(), agent, max_action, None, n_plot_batch) for i_env in env]
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()
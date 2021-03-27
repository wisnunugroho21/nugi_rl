import gym
import random
import numpy as np
import torch
import os
# import ray

from torch.utils.tensorboard import SummaryWriter

from eps_runner.iteration.carla import CarlaRunner
from train_executor.executor import Executor
from agent.ppg_clr import AgentPpgClr
from distribution.basic import BasicContinous
from environment.custom.carla_env import CarlaEnv
from loss.other.joint_aux import JointAux
from loss.ppo.truly_ppo import TrulyPPO
from loss.other.clr import CLR
from policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from model.ppg_clr.carla_Cnn import CnnModel, ProjectionModel, Policy_Model, Value_Model
from memory.policy.policy_memory import PolicyMemory
from memory.aux_ppg.aux_memory import AuxMemory
from memory.clr.clr_memory import ClrMemory 

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = True # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu                 = True
render                  = False # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 495 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 1000000 # How many episode you want to run
n_memory_clr            = 10000
n_update                = 256 # How many episode before you update the Policy 
n_aux_update            = 2
n_saved                 = n_aux_update

policy_kl_range         = 0.03
policy_params           = 5
value_clip              = 4.0
entropy_coef            = 0.0
vf_loss_coef            = 1.0
batch_size              = 32
PPO_epochs              = 4
Aux_epochs              = 4
Clr_epochs              = 2
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 3e-4

folder                  = 'weights/carla'
env                     = CarlaEnv(im_height = 320, im_width = 320, im_preview = False, max_step = 512) # gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]

state_dim           = None
action_dim          = None
max_action          = 1

Policy_Model        = Policy_Model
Value_Model         = Value_Model
Cnn_Model           = CnnModel
ProjectionModel     = ProjectionModel
Policy_Dist         = BasicContinous
Runner              = CarlaRunner
Executor            = Executor
Policy_loss         = TrulyPPO
Aux_loss            = JointAux
Clr_loss            = CLR
Wrapper             = env
Policy_Memory       = PolicyMemory
Aux_Memory          = AuxMemory
Clr_Memory          = ClrMemory
Advantage_Function  = GeneralizedAdvantageEstimation

#####################################################################################################################################################

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)
os.environ['PYTHONHASHSEED'] = str(20)

if state_dim is None:
    state_dim = Wrapper.get_obs_dim()
print('state_dim: ', state_dim)

if Wrapper.is_discrete():
    print('discrete')
else:
    print('continous')

if action_dim is None:
    action_dim = Wrapper.get_action_dim()
print('action_dim: ', action_dim)

policy_dist         = Policy_Dist(use_gpu)
advantage_function  = Advantage_Function(gamma)
aux_memory          = Aux_Memory()
policy_memory       = Policy_Memory()
runner_memory       = Policy_Memory()
clr_memory          = Clr_Memory(n_memory_clr)
aux_loss            = Aux_loss(policy_dist)
policy_loss         = Policy_loss(policy_dist, advantage_function, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma)
clr_loss            = Clr_loss(use_gpu)

""" agent = AgentPPG(Policy_Model, Value_Model, state_dim, action_dim, distribution, policy_loss, aux_loss, policy_memory, aux_memory, 
                PPO_epochs, Aux_epochs, n_aux_update, is_training_mode, policy_kl_range, policy_params, value_clip, 
                entropy_coef, vf_loss_coef, batch_size,  learning_rate, folder, use_gpu) """

agent = AgentPpgClr( Policy_Model, Value_Model, CnnModel, ProjectionModel, state_dim, action_dim, policy_dist, policy_loss, aux_loss, clr_loss, 
                policy_memory, aux_memory, clr_memory, PPO_epochs, Aux_epochs, Clr_epochs, n_aux_update, 
                is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef, 
                batch_size,  learning_rate, folder, use_gpu)

""" agent = AgentPpgVae(Policy_Model, Value_Model, CnnModel, DecoderModel, state_dim, action_dim, policy_dist, policy_loss, aux_loss, vae_loss, 
                policy_memory, aux_memory, vae_memory, PPO_epochs, Aux_epochs, Vae_epochs, n_ae_update, n_aux_update, 
                is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef, 
                batch_size,  learning_rate, folder, use_gpu) """

# ray.init()
runner      = Runner(agent, Wrapper, runner_memory, is_training_mode, render, n_update, Wrapper.is_discrete, max_action, SummaryWriter(), n_plot_batch) # [Runner.remote(i_env, render, training_mode, n_update, Wrapper.is_discrete(), agent, max_action, None, n_plot_batch) for i_env in env]
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()
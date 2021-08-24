import random
import numpy as np
import torch
import os

from torch.utils.tensorboard import SummaryWriter
from torch.optim.adamw import AdamW

from eps_runner.iteration.carla import CarlaRunner
from train_executor.executor import Executor
from agent.image_state.ppg_clr_lstm import AgentImageStatePPGClr
from distribution.basic_continous import BasicContinous
from environment.custom.carla.carla_rgb_timestep import CarlaEnv
from loss.other.aux_ppg import AuxPPG
from loss.ppo.truly_ppo import TrulyPPO
from loss.clr.moco import Moco
from policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from model.ppg.carla_lstm.cnn_model import CnnModel
from model.ppg.carla_lstm.policy_std_model import PolicyModel
from model.ppg.carla_lstm.value_model import ValueModel
from model.ppg.carla_lstm.projection_model import ProjectionModel
from memory.policy.image_state.timestep import TimeImageStatePolicyMemory
from memory.aux_ppg.image_state.timestep import TimeImageStateAuxPpgMemory
from memory.aux_clr.timestep import TimeAuxClrMemory

from helpers.pytorch_utils import set_device

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
n_update                = 512 # How many episode before you update the Policy 
n_aux_update            = 2
n_saved                 = n_aux_update

policy_kl_range         = 0.03
policy_params           = 5
value_clip              = 5.0
entropy_coef            = 0.2
vf_loss_coef            = 1.0
batch_size              = 32
ppo_epochs              = 5
aux_ppg_epochs          = 5
aux_clr_epochs          = 5
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 3e-4

device                  = torch.device('cuda:0')
folder                  = 'weights/carla2'
# env                     = gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]

state_dim           = None
action_dim          = None
max_action          = 1

#####################################################################################################################################################

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)
os.environ['PYTHONHASHSEED'] = str(20)

environment         = CarlaEnv(im_height = 320, im_width = 320, im_preview = False, max_step = 256)

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

policy_dist         = BasicContinous(use_gpu)
advantage_function  = GeneralizedAdvantageEstimation(gamma)

ppo_memory          = TimeImageStatePolicyMemory()
aux_ppg_memory      = TimeImageStateAuxPpgMemory()
aux_clr_memory      = TimeAuxClrMemory()

aux_ppg_loss        = AuxPPG(policy_dist)
ppo_loss            = TrulyPPO(policy_dist, advantage_function, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma)
aux_clr_loss        = Moco(use_gpu)

cnn                 = CnnModel().float().to(set_device(use_gpu))
policy              = PolicyModel(state_dim, action_dim, use_gpu).float().to(set_device(use_gpu))
value               = ValueModel(state_dim).float().to(set_device(use_gpu))
projector           = ProjectionModel().float().to(set_device(use_gpu))

ppo_optimizer       = AdamW(list(policy.parameters()) + list(value.parameters()) + list(cnn.parameters()), lr = learning_rate)        
aux_ppg_optimizer   = AdamW(list(policy.parameters()), lr = learning_rate)
aux_clr_optimizer   = AdamW(list(cnn.parameters()) + list(projector.parameters()), lr = learning_rate)

agent   = AgentImageStatePPGClr(projector, cnn, policy, value, policy_dist, ppo_loss, aux_ppg_loss, aux_clr_loss, ppo_memory, aux_ppg_memory, aux_clr_memory,
            ppo_optimizer, aux_ppg_optimizer, aux_clr_optimizer, ppo_epochs, aux_ppg_epochs, aux_clr_epochs, n_aux_update, is_training_mode, 
            batch_size, folder, use_gpu)

runner      = CarlaRunner(agent, environment, is_training_mode, render, n_update, environment.is_discrete(), max_action, SummaryWriter(), n_plot_batch) # [Runner.remote(i_env, render, training_mode, n_update, Wrapper.is_discrete(), agent, max_action, None, n_plot_batch) for i_env in env]
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()
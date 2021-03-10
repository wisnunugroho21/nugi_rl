import gym
import random
import numpy as np
import torch
import os
# import ray

from torch.utils.tensorboard import SummaryWriter

from eps_runner.iteration.pong_eps_full import PongFullRunner
from train_executor.executor import Executor
from agent.ppg_vae import AgentPpgVae
from distribution.basic import BasicDiscrete
from environment.wrapper.gym_wrapper import GymWrapper
from loss.other.joint_aux import JointAux
from loss.ppo.truly_ppo import TrulyPPO
from loss.other.vae import VAE
from policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from model.ppg_vae.CnnLSTM import CnnModel, DecoderModel, Policy_Model, Value_Model
from memory.policy.policy_memory import PolicyMemory
from memory.aux_ppg.aux_memory import AuxMemory
from memory.vae.image_timestep_vae_memory import ImageTimestepVaeMemory 

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = True # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu                 = True
render                  = True # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 495 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 1000000 # How many episode you want to run
n_memory_vae            = 256
n_update                = 8 # How many episode before you update the Policy
n_ppo_update            = 64 
n_aux_update            = 5 
n_saved                 = n_update * n_ppo_update * n_aux_update

policy_kl_range         = 0.0008
policy_params           = 20
value_clip              = 4.0
entropy_coef            = 0.01
vf_loss_coef            = 1.0
batch_size              = 32
PPO_epochs              = 4
Aux_epochs              = 4
Vae_epochs              = 1
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 3e-4

folder                  = 'weights/pong'
env                     = gym.make('PongNoFrameskip-v4') # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]

state_dim           = 80 * 80
action_dim          = 3
max_action          = 1

Policy_Model        = Policy_Model
Value_Model         = Value_Model
Cnn_Model           = CnnModel
Decoder_Model       = DecoderModel
Policy_Dist         = BasicDiscrete
Runner              = PongFullRunner
Executor            = Executor
Policy_loss         = TrulyPPO
Aux_loss            = JointAux
Vae_loss            = VAE
Wrapper             = GymWrapper(env)
Policy_Memory       = PolicyMemory
Aux_Memory          = AuxMemory
Vae_Memory          = ImageTimestepVaeMemory
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
vae_memory          = Vae_Memory(n_memory_vae)
aux_loss            = Aux_loss(policy_dist)
policy_loss         = Policy_loss(policy_dist, advantage_function, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma)
vae_loss            = Vae_loss()

""" agent = AgentPPG(Policy_Model, Value_Model, state_dim, action_dim, distribution, policy_loss, aux_loss, policy_memory, aux_memory, 
                PPO_epochs, Aux_epochs, n_aux_update, is_training_mode, policy_kl_range, policy_params, value_clip, 
                entropy_coef, vf_loss_coef, batch_size,  learning_rate, folder, use_gpu) """

""" agent = AgentPpgClr(Policy_Model, Value_Model, state_dim, action_dim, distribution, policy_loss, aux_loss, clr_loss, policy_memory, aux_memory, clr_memory, PPO_epochs, Aux_epochs, Clr_epochs, 
            n_ppo_update, n_aux_update, is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef, batch_size,  learning_rate, folder, use_gpu) """

agent = AgentPpgVae(Policy_Model, Value_Model, CnnModel, DecoderModel, state_dim, action_dim, policy_dist, policy_loss, aux_loss, vae_loss, 
                policy_memory, aux_memory, vae_memory, PPO_epochs, Aux_epochs, Vae_epochs, n_ppo_update, n_aux_update, 
                is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef, 
                batch_size,  learning_rate, folder, use_gpu)

# ray.init()
runner      = Runner(agent, Wrapper, policy_memory, is_training_mode, render, n_update, Wrapper.is_discrete, max_action, SummaryWriter(), n_plot_batch) # [Runner.remote(i_env, render, training_mode, n_update, Wrapper.is_discrete(), agent, max_action, None, n_plot_batch) for i_env in env]
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights)

executor.execute()
import gym
import random
import numpy as np
import torch
import os
# import ray

from torch.utils.tensorboard import SummaryWriter

from eps_runner.off_policy.runner import OffRunner
from train_executor.off_policy.executor import OffExecutor
from agent.sac import AgentSAC
from distribution.basic import BasicContinous
from loss.sac.q_loss import QLoss
from loss.sac.v_loss import VLoss
from loss.sac.policy_loss import PolicyLoss
""" from loss.joint_aux import JointAux
from loss.ppo.truly_ppo import TrulyPPO
from loss.clr import CLR
from model.ppg.TanhStdNN import Policy_Model, Value_Model """
from model.sac.TanhStdNN import Q_Model, Value_Model, Policy_Model
from environment.wrapper.gym_wrapper import GymWrapper
from memory.off_policy.off_policy_memory import OffPolicyMemory
""" from memory.on_policy.on_policy_memory import OnPolicyMemory
from memory.other.aux_memory import AuxMemory
from memory.other.clr_memory import ClrMemory """

# from environment.custom.carla_env import CarlaEnv
""" from mlagents_envs.registry import default_registry
from mlagents_envs.environment import UnityEnvironment """
#from gym_unity.envs import UnityToGymWrapper

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = False # If you want to save the agent, set this to True
training_mode           = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu                 = True
render                  = True # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 495 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 1000000 # How many episode you want to run
n_update                = 1
n_saved                 = 100
""" n_memory_clr            = 128
n_update                = 1024 # How many episode before you update the Policy
n_ppo_update            = 1024 
n_aux_update            = 5 
n_saved                 = n_update * n_ppo_update * n_aux_update """

""" policy_kl_range         = 0.03
policy_params           = 5
value_clip              = 10.0
entropy_coef            = 0.01
vf_loss_coef            = 1.0
batch_size              = 32
PPO_epochs              = 10
Aux_epochs              = 10
Clr_epochs              = 5
action_std              = 1.0
gamma                   = 0.99
lam                     = 0.95
learning_rate           = 3e-4 """

epochs                  = 1
batch_size              = 32
soft_tau                = 0.01
gamma                   = 0.99
lam                     = 0.95              
learning_rate           = 3e-4    
folder                  = 'model'
use_gpu                 = True
max_action              = 1.0

folder                  = 'weights/pong'
env                     = gym.make('BipedalWalker-v3') # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]
#env                     = UnityEnvironment(file_name=None, seed=1)
#env                     = UnityToGymWrapper(env)

state_dim       = None
action_dim      = None

""" Policy_Model    = Policy_Model
Value_Model     = Value_Model
Distribution    = BasicContinous
Runner          = Runner
Executor        = Executor
Policy_loss     = TrulyPPO
Aux_loss        = JointAux
Clr_loss        = CLR
Wrapper         = GymWrapper(env) # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512)
Policy_Memory   = OnPolicyMemory
Aux_Memory      = AuxMemory
Clr_Memory      = ClrMemory """

Agent           = AgentSAC
Q_Model         = Q_Model
Value_Model     = Value_Model
Policy_Model    = Policy_Model
Distribution    = BasicContinous
Runner          = OffRunner
Executor        = OffExecutor
Q_Loss          = QLoss
V_Loss          = VLoss
Policy_Loss     = PolicyLoss
Wrapper         = GymWrapper(env)
Memory          = OffPolicyMemory

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

""" distribution    = Distribution(use_gpu)
aux_memory      = Aux_Memory()
policy_memory   = Policy_Memory()
runner_memory   = Policy_Memory()
clr_memory      = Clr_Memory(n_memory_clr)
aux_loss        = Aux_loss(distribution)
policy_loss     = Policy_loss(distribution, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma, lam)
clr_loss        = Clr_loss() """

""" agent = AgentPPG(Policy_Model, Value_Model, state_dim, action_dim, distribution, policy_loss, aux_loss, policy_memory, aux_memory,
     training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef, batch_size, PPO_epochs, Aux_epochs, 
     gamma, lam, learning_rate, folder, use_gpu, n_aux_update)

agent = AgentPpgClr(Policy_Model, Value_Model, state_dim, action_dim, distribution, policy_loss, aux_loss, clr_loss, policy_memory, aux_memory, clr_memory, training_mode, policy_kl_range, 
    policy_params, value_clip, entropy_coef, vf_loss_coef, batch_size, PPO_epochs, Aux_epochs, Clr_epochs, gamma, lam, learning_rate, folder, use_gpu, n_ppo_update, n_aux_update)

# ray.init()
runner      = Runner(Wrapper, render, training_mode, n_update, Wrapper.is_discrete(), runner_memory, agent, max_action, SummaryWriter(), n_plot_batch) # [Runner.remote(i_env, render, training_mode, n_update, Wrapper.is_discrete(), agent, max_action, None, n_plot_batch) for i_env in env]
executor    = Executor(agent, env, n_iteration, runner, reward_threshold, save_weights, n_plot_batch, n_saved, max_action, load_weights)

executor.execute() """

distribution    = Distribution(use_gpu)
memory          = Memory(1000)
q_loss          = Q_Loss(gamma)
v_loss          = V_Loss(distribution)
policy_loss     = Policy_Loss(distribution)

agent       = Agent(Q_Model, Value_Model, Policy_Model, state_dim, action_dim, distribution, q_loss, v_loss, policy_loss, memory, training_mode, 
                batch_size, epochs, soft_tau, gamma , lam , learning_rate, folder, use_gpu)

runner      = Runner(env, render, training_mode, n_update, Wrapper.is_discrete(), memory, agent, max_action)
executor    = Executor(agent, n_iteration, runner, reward_threshold, save_weights, n_plot_batch, n_saved, max_action, SummaryWriter())

executor.execute()
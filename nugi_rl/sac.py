import gym
import random
import numpy as np
import torch
import os

from torch.utils.tensorboard import SummaryWriter
from torch.optim.adamw import AdamW

from eps_runner.single_step.single_step_runner import SingleStepRunner
from train_executor.executor import Executor
from agent.standard.sac import AgentSAC
from distribution.tanh_clipped_continous import TanhClippedContinous
from environment.wrapper.gym_wrapper import GymWrapper
from loss.sac.q_loss import QLoss
from loss.sac.policy_loss import PolicyLoss
from loss.sac.value_loss import ValueLoss
from model.sac.TanhStdNN import Policy_Model, Q_Model, Value_Model
from memory.policy.standard import PolicyMemory

from helpers.pytorch_utils import set_device

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = False # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
is_save_memory          = True
render                  = True # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 495 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_memory                = 102400
n_iteration             = 1000000
n_plot_batch            = 1
soft_tau                = 0.995
n_saved                 = 1
epochs                  = 1
batch_size              = 100
action_std              = 1.0
learning_rate           = 3e-4
alpha                   = 0.2

device                  = torch.device('cuda:0')
folder                  = 'weights/carla'
env                     = gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]

state_dim           = None
action_dim          = None
max_action          = 1

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

distribution        = TanhClippedContinous()
memory              = PolicyMemory(capacity = n_memory)

q_loss              = QLoss(distribution)
policy_loss         = PolicyLoss(distribution, alpha = alpha)
value_loss          = ValueLoss(distribution, alpha = alpha)

policy              = Policy_Model(state_dim, action_dim).float().to(device)
soft_q1             = Q_Model(state_dim, action_dim).float().to(device)
soft_q2             = Q_Model(state_dim, action_dim).float().to(device)
value               = Value_Model(state_dim).float().to(device)

policy_optimizer    = AdamW(list(policy.parameters()), lr = learning_rate)        
soft_q_optimizer    = AdamW(list(soft_q1.parameters()) + list(soft_q2.parameters()), lr = learning_rate)
value_optimizer     = AdamW(list(value.parameters()), lr = learning_rate)

agent = AgentSAC(soft_q1, soft_q2, policy, value, distribution, q_loss, policy_loss, value_loss, memory, 
        soft_q_optimizer, policy_optimizer, value_optimizer, is_training_mode, batch_size, epochs, 
        soft_tau, folder, device)
                    
runner      = SingleStepRunner(agent, environment, is_save_memory, render, environment.is_discrete(), max_action, SummaryWriter(), n_plot_batch) # [Runner.remote(i_env, render, training_mode, n_update, Wrapper.is_discrete(), agent, max_action, None, n_plot_batch) for i_env in env]
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()
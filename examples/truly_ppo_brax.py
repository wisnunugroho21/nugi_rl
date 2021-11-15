import gym
import random
import numpy as np
import torch
import os

from torch.utils.tensorboard import SummaryWriter
from torch.optim.adamw import AdamW

from nugi_rl.train.runner.iteration.brax import BraxIterRunner
from nugi_rl.train.executor.standard import Executor
from nugi_rl.agent.ppo import AgentPPO
from nugi_rl.distribution.continous.basic_continous import BasicContinous
from nugi_rl.environment.wrapper.brax_wrapper import BraxWrapper
from nugi_rl.loss.ppo.truly_ppo import TrulyPpo
from nugi_rl.loss.value import ValueLoss
from nugi_rl.loss.entropy import EntropyLoss
from nugi_rl.policy_function.advantage_function.gae import GeneralizedAdvantageEstimation
from nugi_rl.model.ppo.TanhStdNN import Policy_Model, Value_Model
from nugi_rl.memory.policy.standard import PolicyMemory

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = False # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
render                  = False # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 1000 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_iteration             = 100000000 # How many episode you want to run
n_update                = 1024 # How many episode before you update the Policy 
n_saved                 = 1

policy_kl_range         = 0.03
policy_params           = 5
value_clip              = None
entropy_coef            = 0.01
vf_loss_coef            = 1.0
batch_size              = 3084
epochs                  = 5
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 3e-4
bins                    = 9

device                  = torch.device('cuda')
folder                  = 'weights/truly_ppo_lift1'

BraxWrapper.register_brax_gym(gym)
env                     = gym.make('brax_ant-v0', batch_size = 96)

state_dim               = None
action_dim              = None
max_action              = 1

#####################################################################################################################################################

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)
os.environ['PYTHONHASHSEED'] = str(20)

environment         = BraxWrapper(env)

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

memory          = PolicyMemory()
ppo_loss        = TrulyPpo(distribution, policy_kl_range, policy_params)
value_loss      = ValueLoss(vf_loss_coef, value_clip)
entropy_loss    = EntropyLoss(distribution, entropy_coef)

policy          = Policy_Model(state_dim, action_dim).float().to(device)
value           = Value_Model(state_dim).float().to(device)
optimizer       = AdamW(list(policy.parameters()) + list(value.parameters()), lr = learning_rate)

agent   = AgentPPO(policy, value, advantage_function, distribution, ppo_loss, value_loss, entropy_loss, memory, optimizer, 
    epochs, is_training_mode, batch_size, folder, device)

runner      = BraxIterRunner(agent, environment, is_training_mode, render, n_update, SummaryWriter())
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()
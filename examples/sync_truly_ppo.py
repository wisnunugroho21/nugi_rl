import gym
import random
import numpy as np
import torch
import os
import ray
import robosuite as suite
from robosuite.wrappers import GymWrapper as RobosuiteWrapper

from torch.utils.tensorboard import SummaryWriter
from torch.optim.adamw import AdamW

from nugi_rl.train.runner.iteration.sync import SyncRunner
from nugi_rl.train.executor.sync import SyncExecutor
from nugi_rl.agent.ppo import AgentPPO
from nugi_rl.distribution.discrete.ordinal import Ordinal
from nugi_rl.environment.wrapper.discretization_gym_wrapper import DiscretizationGymWrapper
from nugi_rl.loss.ppo.truly_ppo import TrulyPpo
from nugi_rl.loss.value import ValueLoss
from nugi_rl.loss.entropy import EntropyLoss
from nugi_rl.policy_function.advantage_function.gae import GeneralizedAdvantageEstimation
from nugi_rl.model.ppo.SigmoidNN import Policy_Model, Value_Model
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

policy_kl_range         = 0.03
policy_params           = 5
value_clip              = None
entropy_coef            = 0.1
vf_loss_coef            = 1.0
batch_size              = 32
epochs                  = 10
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 3e-4
bins                    = 11
n_agents                = 2

device                  = torch.device('cuda')
folder                  = 'weights/bipedal_walker'

# environments            =  [
#     DiscretizationGymWrapper(
#         RobosuiteWrapper(
#             suite.make(
#                 "Lift",
#                 robots = "Sawyer",                # use Sawyer robot
#                 use_camera_obs = False,           # do not use pixel observations
#                 has_offscreen_renderer = False,   # not needed since not using pixel obs
#                 has_renderer = True,              # make sure we can render to the screen
#                 reward_shaping = True,            # use dense rewards
#                 control_freq = 20,                # control should happen fast enough so that simulation looks smooth
#             )
#         )        
#         , bins)
    
#     for _ in range(n_agents)
# ]

environments            =  [
    DiscretizationGymWrapper(
        gym.make('BipedalWalker-v3'),
        torch.device('cpu'),
        bins
    )
    
    for _ in range(n_agents)
]

state_dim               = None
action_dim              = None
max_action              = 1

#####################################################################################################################################################

random.seed(30)
np.random.seed(30)
torch.manual_seed(30)
os.environ['PYTHONHASHSEED'] = str(30)

if state_dim is None:
    state_dim = environments[0].get_obs_dim()
print('state_dim: ', state_dim)

if environments[0].is_discrete():
    print('discrete')
else:
    print('continous')

if action_dim is None:
    action_dim = environments[0].get_action_dim()
print('action_dim: ', action_dim)

distribution        = Ordinal(device)
advantage_function  = GeneralizedAdvantageEstimation(gamma)

agents_worker       = [
    ray.put(
        AgentPPO(Policy_Model(state_dim, action_dim, bins), Value_Model(state_dim), None, Ordinal(torch.device('cpu')), None, None, None, PolicyMemory(torch.device('cpu')), None, 
            None, is_training_mode, None, folder, torch.device('cpu'))
    ) 
    for _ in range(n_agents)
]

environments    = [ray.put(environment) for environment in environments]

runners         = [
    SyncRunner.remote(
        agent, environment, is_training_mode, render, n_update, None, n_plot_batch, idx
    ) 
    for idx, (agent, environment) in enumerate(zip(agents_worker, environments))
]

memory          = PolicyMemory(device)
ppo_loss        = TrulyPpo(distribution, policy_kl_range, policy_params)
value_loss      = ValueLoss(vf_loss_coef, value_clip)
entropy_loss    = EntropyLoss(distribution, entropy_coef)

policy          = Policy_Model(state_dim, action_dim, bins).float().to(device)
value           = Value_Model(state_dim).float().to(device)
optimizer       = AdamW(list(policy.parameters()) + list(value.parameters()), lr = learning_rate)

agent_learner   = AgentPPO(policy, value, advantage_function, distribution, ppo_loss, value_loss, entropy_loss, memory, optimizer, 
    epochs, is_training_mode, batch_size, folder, device)

executor    = SyncExecutor(agent_learner, n_iteration, runners, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()
import gym
import random
import numpy as np
import torch
import os
import redis

from torch.utils.tensorboard import SummaryWriter
from torch.optim.adam import Adam

from eps_runner.single_step.single_step_runner import SingleStepRunner
from train_executor.multi_agent_central_learner.multi_process.central_learner import CentralLearnerExecutor
from agent.standard.cql import AgentCql
from distribution.basic_continous import BasicContinous
from environment.wrapper.gym_wrapper import GymWrapper
from loss.cql.cql import Cql
from loss.cql.policy import OffPolicyLoss
from model.cql.TanhNN import Policy_Model, Q_Model
from memory.policy.redis_list import PolicyRedisListMemory

from helpers.pytorch_utils import set_device

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = False # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
use_gpu                 = True
render                  = True # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 495 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_iteration             = 1000000
n_plot_batch            = 1
soft_tau                = 0.99
n_saved                 = 1
epochs                  = 1
batch_size              = 32
action_std              = 1.0
learning_rate           = 3e-4

folder                  = 'weights/carla'
env                     = gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]

state_dim           = None
action_dim          = None
max_action          = 1

Policy_Model        = Policy_Model
Q_Model             = Q_Model
Policy_Dist         = BasicContinous
Runner              = SingleStepRunner
Executor            = CentralLearnerExecutor
Policy_loss         = OffPolicyLoss
Q_loss              = Cql
Wrapper             = GymWrapper
Policy_Memory       = PolicyRedisListMemory
Agent               = AgentCql

#####################################################################################################################################################

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)
os.environ['PYTHONHASHSEED'] = str(20)

environment = Wrapper(env)

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

redis_obj           = redis.Redis()

policy_dist         = Policy_Dist(use_gpu)
agent_memory        = Policy_Memory(redis_obj)
runner_memory       = Policy_Memory(redis_obj)
executor_memory     = Policy_Memory(redis_obj)
q_loss              = Q_loss()
policy_loss         = Policy_loss()

policy              = Policy_Model(state_dim, action_dim, use_gpu).float().to(set_device(use_gpu))
soft_q1             = Q_Model(state_dim, action_dim).float().to(set_device(use_gpu))
soft_q2             = Q_Model(state_dim, action_dim).float().to(set_device(use_gpu))
policy_optimizer    = Adam(list(policy.parameters()), lr = learning_rate)        
soft_q_optimizer    = Adam(list(soft_q1.parameters()) + list(soft_q2.parameters()), lr = learning_rate)

agent = Agent(soft_q1, soft_q2, policy, state_dim, action_dim, q_loss, policy_loss, agent_memory, soft_q_optimizer, 
        policy_optimizer, is_training_mode, batch_size, epochs, folder, use_gpu)
                    
runner      = Runner(agent, environment, runner_memory, is_training_mode, render, environment.is_discrete(), max_action, SummaryWriter(), n_plot_batch) # [Runner.remote(i_env, render, training_mode, n_update, Wrapper.is_discrete(), agent, max_action, None, n_plot_batch) for i_env in env]
executor    = Executor(agent, n_iteration, executor_memory, runner, save_weights, n_saved) 

executor.execute()
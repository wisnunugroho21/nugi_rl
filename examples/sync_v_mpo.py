import gym
import random
import numpy as np
import torch
import os
import pybullet_envs
import ray
# from pybullet_envs.deep_mimic.gym_env.deep_mimic_env import HumanoidDeepMimicWalkBulletEnv

from torch.optim.adamw import AdamW

from eps_runner.iteration.sync import SyncRunner
from train_executor.sync import SyncExecutor
from agent.standard.v_mpo import AgentVMPO
from distribution.multivariate_continous import MultivariateContinous
from environment.wrapper.gym_wrapper import GymWrapper
from loss.v_mpo.phi_loss import PhiLoss
from loss.v_mpo.temperature_loss import TemperatureLoss
from loss.v_mpo.continous_alpha_loss import AlphaLoss
from loss.v_mpo.value_loss import ValueLoss
from loss.v_mpo.entropy_loss import EntropyLoss
from policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from model.v_mpo.TanhNN import Policy_Model, Value_Model
from memory.policy.standard import PolicyMemory

############## Hyperparameters ##############

load_weights            = True # If you want to load the agent, set this to True
save_weights            = True # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
render                  = True # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 1000 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 100000000 # How many episode you want to run
n_update                = 1024 # How many episode before you update the Policy
n_saved                 = 1
n_agents                = 4

coef_alpha_mean_upper   = 0.01
coef_alpha_mean_below   = 0.005
coef_alpha_cov_upper    = 5e-5
coef_alpha_cov_below    = 5e-6

# coef_alpha_upper        = 0.01
# coef_alpha_below        = 0.005

coef_temp               = 0.01
batch_size              = 256
policy_epochs           = 5
value_clip              = None
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 1e-4
entropy_coef            = 0.1

device                  = torch.device('cuda')
folder                  = 'weights/v_mpo_humanoid'

environments            =  [GymWrapper(gym.make('HumanoidBulletEnv-v0')) for _ in range(n_agents)]

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

coef_alpha_mean_upper   = torch.Tensor([coef_alpha_mean_upper]).to(device)
coef_alpha_mean_below   = torch.Tensor([coef_alpha_mean_below]).to(device)
coef_alpha_cov_upper    = torch.Tensor([coef_alpha_cov_upper]).to(device)
coef_alpha_cov_below    = torch.Tensor([coef_alpha_cov_below]).to(device)

# coef_alpha_upper    = torch.Tensor([coef_alpha_upper]).to(device)
# coef_alpha_below    = torch.Tensor([coef_alpha_below]).to(device)

distribution        = MultivariateContinous()
advantage_function  = GeneralizedAdvantageEstimation(gamma)
policy_memory       = PolicyMemory()

agents_worker       = [
    ray.put(            
        AgentVMPO(Policy_Model(state_dim, action_dim).float(), Value_Model(state_dim).float(), 
            distribution, None, None, None, None, None,
            PolicyMemory(), None, None, None, is_training_mode, None, folder, 
            torch.device('cpu'))
    ) 
    for _ in range(n_agents)
]

environments    = [ray.put(environment) for environment in environments]

runners         = [
    SyncRunner.remote(
        agent, environment, is_training_mode, render, n_update, False, max_action, None, n_plot_batch, idx
    ) 
    for idx, (agent, environment) in enumerate(zip(agents_worker, environments))
]

learner_policy      = Policy_Model(state_dim, action_dim).float().to(device)
learner_value       = Value_Model(state_dim).float().to(device)
policy_optimizer    = AdamW(learner_policy.parameters(), lr = learning_rate)
value_optimizer     = AdamW(learner_value.parameters(), lr = learning_rate)

alpha_loss          = AlphaLoss(distribution, coef_alpha_mean_upper, coef_alpha_mean_below, coef_alpha_cov_upper, coef_alpha_cov_below)
phi_loss            = PhiLoss(distribution, advantage_function)
temperature_loss    = TemperatureLoss(advantage_function, coef_temp, device)
value_loss          = ValueLoss(advantage_function, value_clip)
entropy_loss        = EntropyLoss(distribution, entropy_coef)

agent_learner = AgentVMPO(learner_policy, learner_value, distribution, alpha_loss, phi_loss, entropy_loss, temperature_loss, value_loss,
                    policy_memory, policy_optimizer, value_optimizer, policy_epochs, is_training_mode, batch_size, folder, 
                    device)

executor    = SyncExecutor(agent_learner, n_iteration, runners, save_weights, n_saved, load_weights, is_training_mode)
executor.execute()
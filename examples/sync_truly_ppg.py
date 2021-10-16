import gym
import random
import numpy as np
import torch
import os
import ray
from pybullet_envs.deep_mimic.gym_env.deep_mimic_env import HumanoidDeepMimicWalkBulletEnv

from torch.optim.adamw import AdamW

from nugi_rl.eps_runner.iteration.sync import SyncRunner
from nugi_rl.train_executor.sync import SyncExecutor
from nugi_rl.agent.standard.ppg import AgentPPG
from nugi_rl.distribution.basic_continous import BasicContinous
from nugi_rl.environment.wrapper.gym_wrapper import GymWrapper
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
n_aux_update            = 10
n_saved                 = n_aux_update

policy_kl_range         = 0.05
policy_params           = 5
value_clip              = 5.0
entropy_coef            = 0.1
vf_loss_coef            = 1.0
batch_size              = 64
ppo_epochs              = 10
aux_ppg_epochs          = 10
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 3e-4
n_agents                = 4

device                  = torch.device('cuda:0')
folder                  = 'weights/bipedal'

environments            =  [GymWrapper(gym.make('HumanoidBulletEnv-v0')) for _ in range(n_agents)]
# env                     = HumanoidDeepMimicWalkBulletEnv(renders = render) # gym.make('BipedalWalker-v3') # HumanoidDeepMimicWalkBulletEnv(renders = render) # gym.make("KukaBulletEnv-v0") # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]

state_dim               = None
action_dim              = None
max_action              = 1

#####################################################################################################################################################

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)
os.environ['PYTHONHASHSEED'] = str(20)

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

environments        = [ray.put(environment) for environment in environments]

distribution        = BasicContinous()
advantage_function  = GeneralizedAdvantageEstimation(gamma)

aux_ppg_loss        = AuxPPG(distribution)
ppo_loss            = TrulyPPO(distribution, advantage_function, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef)

agents_worker       = [
    ray.put(
        AgentPPG(Policy_Model(state_dim, action_dim).float(), Value_Model(state_dim).float(), distribution, ppo_loss, aux_ppg_loss, PolicyMemory(), AuxPpgMemory(), 
            None, None, ppo_epochs, aux_ppg_epochs, n_aux_update, is_training_mode, 
            batch_size,  folder, torch.device('cpu'))
    ) 
    for _ in range(n_agents)
]
 
runners         = [
    SyncRunner.remote(
        agent, environment, is_training_mode, render, n_update, False, max_action, None, n_plot_batch, idx
    ) 
    for idx, (agent, environment) in enumerate(zip(agents_worker, environments))
]

learner_policy      = Policy_Model(state_dim, action_dim).float().to(device)
learner_value       = Value_Model(state_dim).float().to(device)
ppo_optimizer       = AdamW(list(learner_policy.parameters()) + list(learner_value.parameters()), lr = learning_rate)        
aux_ppg_optimizer   = AdamW(list(learner_policy.parameters()), lr = learning_rate)

agent_learner = AgentPPG(learner_policy, learner_value, distribution, ppo_loss, aux_ppg_loss, PolicyMemory(), AuxPpgMemory(), 
    ppo_optimizer, aux_ppg_optimizer, ppo_epochs, aux_ppg_epochs, n_aux_update, is_training_mode, 
    batch_size,  folder, device)

executor    = SyncExecutor(agent_learner, n_iteration, runners, save_weights, n_saved, load_weights, is_training_mode)
executor.execute()
import gym
import random
import numpy as np
import torch
import os
import pybullet_envs

from gym.envs.registration import register

from torch.utils.tensorboard import SummaryWriter
from torch.optim.adamw import AdamW

from nugi_rl.eps_runner.single_step.frozenlake import FrozenlakeSingleStepRunner
from nugi_rl.train_executor.episodic_iter import EpisodicIterExecutor
from nugi_rl.agent.standard.ppo_rnd import AgentPpoRnd
from nugi_rl.distribution.basic_discrete import BasicDiscrete
from nugi_rl.environment.wrapper.gym_wrapper import GymWrapper
from nugi_rl.loss.rnd.truly_ppo import TrulyPPO
from nugi_rl.loss.rnd.state_predictor import RndStatePredictor
from nugi_rl.policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from nugi_rl.model.ppo_rnd.SoftmaxNN import Policy_Model, Value_Model, RND_Model
from nugi_rl.memory.policy.standard import PolicyMemory
from nugi_rl.memory.obs.rnd import RndMemory

try:
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.8196, # optimum = .8196
    )

    print('Env FrozenLakeNotSlippery has not yet initialized. \nInitializing now...')
except:
    print('Env FrozenLakeNotSlippery has been initialized')


############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = False # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
render                  = False # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 1000 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 100000000 # How many episode you want to run
n_update_iter           = 32 # How many episode before you update the Policy
n_update_episodic       = 5
n_saved                 = 1

policy_kl_range         = 0.0008
policy_params           = 20
value_clip              = None
entropy_coef            = 0.05
vf_loss_coef            = 1.0
batch_size              = 32
ppo_epochs              = 4
rnd_epochs              = 4
action_std              = 1.0
clip_norm               = 5
gamma                   = 0.95
learning_rate           = 2.5e-4
ex_advantages_coef      = 2
in_advantages_coef      = 1

device                  = torch.device('cuda:0')
folder                  = 'weights/truly_ppg_humanoid'

env                     = gym.make('FrozenLakeNotSlippery-v0') #gym.make("HumanoidBulletEnv-v0") # gym.make('BipedalWalker-v3') for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, max_step = 512) # [gym.make(env_name) for _ in range(2)] # CarlaEnv(im_height = 240, im_width = 240, im_preview = False, seconds_per_episode = 3 * 60) # [gym.make(env_name) for _ in range(2)] # gym.make(env_name) # [gym.make(env_name) for _ in range(2)]

state_dim               = None
action_dim              = None
max_action              = 1

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

distribution        = BasicDiscrete()
advantage_function  = GeneralizedAdvantageEstimation(gamma)

ppo_memory          = PolicyMemory()
rnd_memory          = RndMemory(state_dim)

ppo_loss            = TrulyPPO(distribution, advantage_function, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, ex_advantages_coef, in_advantages_coef)
rnd_predictor_loss  = RndStatePredictor()

policy              = Policy_Model(state_dim, action_dim).float().to(device)
ex_value            = Value_Model(state_dim).float().to(device)
in_value            = Value_Model(state_dim).float().to(device)
rnd_predict         = RND_Model(state_dim).float().to(device)
rnd_target          = RND_Model(state_dim).float().to(device)

ppo_optimizer       = AdamW(list(policy.parameters()) + list(ex_value.parameters()) + list(in_value.parameters()), lr = learning_rate)
rnd_optimizer       = AdamW(list(rnd_predict.parameters()), lr = learning_rate)

agent   = AgentPpoRnd(policy, ex_value, in_value, rnd_predict, rnd_target, distribution, ppo_loss, rnd_predictor_loss, ppo_memory, rnd_memory, ppo_optimizer, rnd_optimizer, 
            ppo_epochs, rnd_epochs, is_training_mode, clip_norm, batch_size,  folder, device)

policy      = Policy_Model(state_dim, action_dim).float().to(device)
runner      = FrozenlakeSingleStepRunner(state_dim, agent, env, is_training_mode, render, environment.is_discrete(), max_action, SummaryWriter(), n_plot_batch) # [Runner.remote(i_env, render, training_mode, n_update, Wrapper.is_discrete(), agent, max_action, None, n_plot_batch) for i_env in env]
executor    = EpisodicIterExecutor(agent, n_iteration, n_update_iter, n_update_episodic, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()
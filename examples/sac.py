import gym
import torch

from torch.optim.adamw import AdamW

from nugi_rl.train.runner.single_step.standard import SingleStepRunner
from nugi_rl.train.executor.standard import Executor
from nugi_rl.agent.sac import AgentSac
from nugi_rl.distribution.continous.basic import BasicContinous
from nugi_rl.environment.wrapper.gym_wrapper import GymWrapper
from nugi_rl.loss.sac.policy_loss import PolicyLoss
from nugi_rl.loss.sac.q_loss import QLoss
from nugi_rl.model.sac.TanhStdNN import Policy_Model, Q_Model
from nugi_rl.memory.policy.standard import PolicyMemory
from nugi_rl.helpers.plotter.weight_bias import WeightBiasPlotter

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = False # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
render                  = True # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 1000 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 100000000 # How many episode you want to run
n_update                = 1024 # How many episode before you update the Policy 
n_saved                 = 1 # How many iteration before you save the model

alpha                   = 0.2 # Coefficient that control how matters log_prob in policy loss
soft_tau                = 0.95 # the soft update coefficient (polyak update)
batch_size              = 32 # The size of batch for each update
epochs                  = 5 # The amount of epochs for each update
learning_rate           = 3e-4

device_name             = 'cuda'
env_name                = 'BipedalWalker-v3'
folder                  = 'weights'

state_dim               = None
action_dim              = None
max_action              = None

config = { 
    load_weights, save_weights, is_training_mode, render, reward_threshold, n_plot_batch, n_iteration,
    n_update, n_saved, alpha, soft_tau, batch_size,
    epochs, learning_rate, device_name, env_name
}

#####################################################################################################################################################

device              = torch.device(device_name)
environment         = GymWrapper(gym.make(env_name), device)

if state_dim is None:
    state_dim = environment.get_obs_dim()

if action_dim is None:
    action_dim = environment.get_action_dim()

distribution        = BasicContinous()
plotter             = WeightBiasPlotter(config, 'BipedalWalker_v1', entity = "wisnunugroho21")

memory              = PolicyMemory()
q_loss              = QLoss(distribution)
policy_loss         = PolicyLoss(distribution, alpha)

policy              = Policy_Model(state_dim, action_dim).float().to(device)
soft_q1             = Q_Model(state_dim, action_dim).float().to(device)
soft_q2             = Q_Model(state_dim, action_dim).float().to(device)

policy_optimizer    = AdamW(list(policy.parameters()), lr = learning_rate)        
soft_q_optimizer    = AdamW(list(soft_q1.parameters()) + list(soft_q2.parameters()), lr = learning_rate)

agent   = AgentSac(soft_q1, soft_q2, policy, distribution, q_loss, policy_loss, memory, soft_q_optimizer, policy_optimizer,
    is_training_mode, batch_size, epochs, soft_tau, folder, device)

runner      = SingleStepRunner(agent, environment, is_training_mode, render, plotter, n_plot_batch)
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()
import random
import numpy as np
import torch
import os

def run(Runner, Executor, AgentDiscrete, AgentContinous, Policy_or_Actor_Model, Value_or_Critic_Model, env, state_dim, action_dim,
    load_weights, save_weights, training_mode, use_gpu, reward_threshold, render, n_saved, n_plot_batch, n_episode, n_update, n_aux_update,
    policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef, batch_size, PPO_epochs, Aux_epochs, action_std, gamma, lam, learning_rate,
    max_action, folder):

    random.seed(20)    
    np.random.seed(20)
    torch.manual_seed(20)
    os.environ['PYTHONHASHSEED'] = str(20)

    if state_dim is None:
        state_dim = env.get_obs_dim()
    print('state_dim: ', state_dim)

    if action_dim is None:
        action_dim = env.get_action_dim()
    print('action_dim: ', action_dim)

    if env.is_discrete():  
        print('discrete')
        agent = AgentDiscrete(Policy_or_Actor_Model, Value_or_Critic_Model, state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                batch_size, PPO_epochs, Aux_epochs, gamma, lam, learning_rate, folder, use_gpu)

        if load_weights:
            agent.load_weights()
            print('Weight Loaded')

        executor = Executor(agent, env, n_episode, Runner, reward_threshold, save_weights, n_plot_batch, render, training_mode, n_update, n_aux_update, n_saved)
        executor.execute_discrete()

    else:
        print('continous')
        agent = AgentContinous(Policy_or_Actor_Model, Value_or_Critic_Model, state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                batch_size, PPO_epochs, Aux_epochs, gamma, lam, learning_rate, action_std, folder, use_gpu)

        if load_weights:
            agent.load_weights()
            print('Weight Loaded')

        executor = Executor(agent, env, n_episode, Runner, reward_threshold, save_weights, n_plot_batch, render, training_mode, n_update, n_aux_update, n_saved, max_action)
        executor.execute_continous()
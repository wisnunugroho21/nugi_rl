import gym

def run(Runner, Executor, AgentDiscrete, AgentContinous, Policy_or_Actor_Model, Value_or_Critic_Model, env, state_dim, action_dim,
    load_weights, save_weights, training_mode, use_gpu, reward_threshold, render, n_saved, n_plot_batch, n_episode, n_update, n_aux_update,
    policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef, batch_size, PPO_epochs, Aux_epochs, action_std, gamma, lam, learning_rate,
    params_max, params_min, params_subtract, params_dynamic, max_action, folder, use_ppg):    

    if state_dim is None:
        if type(env.observation_space) is gym.spaces.Box:
            if len(env.observation_space.shape) > 1:
                state_dim = 1
                for i in range(len(env.observation_space.shape)):
                    state_dim *= env.observation_space.shape[i]
            
            else:
                state_dim = env.observation_space.shape[0]
                    
        else:
            state_dim = env.observation_space.n

    print('state_dim: ', state_dim)

    if type(env.action_space) is gym.spaces.Box:
        if action_dim is None:
            action_dim = env.action_space.shape[0]
        print('action_dim: ', action_dim)
        print('continous')        

        if use_ppg:
            agent = AgentContinous(Policy_or_Actor_Model, Value_or_Critic_Model, state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                    batch_size, PPO_epochs, Aux_epochs, gamma, lam, learning_rate, action_std, folder, use_gpu)
        else:
            agent = AgentContinous(Policy_or_Actor_Model, Value_or_Critic_Model, state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                    batch_size, PPO_epochs, gamma, lam, learning_rate, action_std, folder, use_gpu)        

        if load_weights:
            agent.load_weights()
            print('Weight Loaded')

        if use_ppg:
            executor = Executor(agent, env, n_episode, Runner, reward_threshold, save_weights, n_plot_batch, render, training_mode, n_update, n_aux_update, 
                n_saved, params_max, params_min, params_subtract, params_dynamic, max_action)

            executor.execute_continous()

    else:
        if action_dim is None:
            action_dim = env.action_space.n
        print('action_dim: ', action_dim)
        print('discrete')        

        if use_ppg:
            agent = AgentDiscrete(Policy_or_Actor_Model, Value_or_Critic_Model, state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                    batch_size, PPO_epochs, Aux_epochs, gamma, lam, learning_rate, folder, use_gpu)
        else:
            agent = AgentDiscrete(Policy_or_Actor_Model, Value_or_Critic_Model, state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                    batch_size, PPO_epochs, gamma, lam, learning_rate, folder, use_gpu)

        if load_weights:
            agent.load_weights()
            print('Weight Loaded')

        if use_ppg:
            executor = Executor(agent, env, n_episode, Runner, reward_threshold, save_weights, n_plot_batch, render, training_mode, n_update, n_aux_update, 
                n_saved, params_max, params_min, params_subtract, params_dynamic, max_action)

            executor.execute_discrete()
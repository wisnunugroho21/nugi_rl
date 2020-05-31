import numpy as np
from utils.math_function import prepo_full_one_dim

def run_discrete_episode(env, agent, render, training_mode, t_updates, n_update, params, params_max, params_min, params_subtract, params_dynamic):
    state       = np.zeros((2, 25600))
    next_state  = np.zeros((2, 25600))
    ############################################
    obs         = env.reset()
    obs         = np.array(prepo_full_one_dim(obs))
    state[1]    = obs

    done            = False
    total_reward    = 0
    eps_time        = 0
    ############################################    
    agent.set_params(params) 
    ############################################
    for _ in range(100000): 
        action      = int(agent.act(state)) 
        action_gym = action + 1 if action != 0 else 0

        next_obs, reward, done, _   = env.step(action_gym)
        next_obs                    = np.array(prepo_full_one_dim(next_obs))
        next_state[0]               = obs
        next_state[1]               = next_obs        

        eps_time += 1 
        t_updates += 1
        total_reward += reward
          
        if training_mode:  
            agent.save_eps(state.tolist(), action, reward, float(done), next_state.tolist()) 
            
        state   = next_state
        obs     = next_obs
                
        if render:
            env.render()     
        
        if training_mode:
            if t_updates == n_update:
                agent.update_ppo()
                t_updates = 0

                if params_dynamic:
                    params = params - params_subtract
                    params = params if params > params_min else params_min
        
        if done: 
            break                

    return total_reward, eps_time, t_updates, params
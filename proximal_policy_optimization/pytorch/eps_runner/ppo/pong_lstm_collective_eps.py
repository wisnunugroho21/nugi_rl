import numpy as np
from utils.math_function import prepro_half_one_dim

def run_discrete_episode(env, agent, render, training_mode, t_updates, n_update, params, params_max, params_min, params_subtract, params_dynamic):
    frame = 4
    ############################################
    state       = np.zeros((frame, 6400))
    next_state  = np.zeros((frame, 6400))
    ############################################
    obs                 = env.reset()
    obs                 = prepro_half_one_dim(obs).reshape(1, 6400)
    state[frame - 1]    = obs

    done            = False
    total_reward    = 0
    eps_time        = 0
    ############################################    
    agent.set_params(params) 
    ############################################
    for _ in range(100000): 
        action      = int(agent.act(state)) 
        action_gym  = action + 1 if action != 0 else 0
        reward      = 0

        for i in range(frame):
            #next_obs, reward, done, _        = env.step(action_gym) if i == 0 else env.step(0)
            
            next_obs, reward_temp, done, _  = env.step(action_gym)
            next_obs                        = prepro_half_one_dim(next_obs).reshape(1, 6400)

            reward      += reward_temp
            next_state  = next_obs if i == 0 else np.concatenate((next_state, next_obs), axis = 0)

            if done:
                if len(next_state) < frame:
                  next_obs      = np.zeros((frame - len(next_state), 6400))
                  next_state    = np.concatenate((next_state, next_obs), axis = 0)  
                break        

        eps_time        += 1 
        t_updates       += 1
        total_reward    += reward
          
        if training_mode:  
            agent.memory.save_eps(state.tolist(), action, reward, float(done), next_state.tolist()) 
            
        state   = next_state
                
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
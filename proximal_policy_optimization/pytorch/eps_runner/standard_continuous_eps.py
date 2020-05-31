import numpy as np

def run_continous_episode(env, agent, render, training_mode, t_updates, n_update, params, params_max, params_min, params_subtract, params_dynamic, max_action):
    ############################################
    state = env.reset()    
    done = False
    total_reward = 0
    eps_time = 0
    ############################################ 
    for _ in range(1, 5000): 
        agent.set_params(params)
        action = agent.act(state)          
        next_state, reward, done, _ = env.step(action)

        eps_time += 1 
        t_updates += 1
        total_reward += reward
          
        if training_mode:
            agent.memory.save_eps(state.tolist(), action.tolist(), reward, float(done), next_state.tolist()) 
            
        state = next_state
                
        if render:
            env.render()     
        
        if training_mode:
            if t_updates == n_update:
                agent.update_ppo()
                t_updates = 0

                if params_dynamic:
                    params = params - params_subtract
                    params = params_min if params < params_min else params
        
        if done: 
            break                

    return total_reward, eps_time, t_updates, params
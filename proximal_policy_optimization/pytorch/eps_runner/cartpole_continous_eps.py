import numpy as np

def run_continous_episode(env, agent, render, training_mode, t_updates, n_update, params, params_max, params_min, params_subtract, params_dynamic, max_action):
    ############################################
    state = env.reset()    
    done = False
    total_reward = 0
    eps_time = 0
    ############################################    
    agent.set_params(params) 
    ############################################
    for i in range(200): 
        action = agent.act(state).numpy()         
        if np.isscalar(action):
            action = np.array([action])
        action_env = np.clip(action, -1, 1) * max_action
        
        next_state, reward, done, _ = env.step(action_env)

        eps_time += 1 
        t_updates += 1
        total_reward += reward
        reward = -10 if i < 200 and done else reward
          
        if training_mode: 
            agent.save_eps(state.tolist(), reward, action, float(done), next_state.tolist()) 
            
        state = next_state
                
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
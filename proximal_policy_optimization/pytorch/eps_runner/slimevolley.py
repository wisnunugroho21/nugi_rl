import numpy as np

def run_discrete_episode(env, agent, render, training_mode, t_updates, n_update, params, params_max, params_min, params_subtract, params_dynamic):
    ############################################
    state = env.reset()    
    done = False
    total_reward = 0
    eps_time = 0
    ############################################    
    agent.set_params(params) 
    ############################################
    for _ in range(10000): 
        action = int(agent.act(state))

        if action == 0:
            action_gym = [0, 0, 0] # NOOP
        elif action == 1:
            action_gym = [1, 0, 0] # LEFT (forward)
        elif action == 2:
            action_gym = [0, 1, 0] # RIGHT (backward)
        elif action == 3:
            action_gym = [0, 0, 1] # UP (jump)
        elif action == 4:
            action_gym = [1, 0, 1] # UPLEFT (forward jump)
        elif action == 5:
            action_gym = [0, 1, 1] # UPRIGHT (backward jump)

        next_state, reward, done, _ = env.step(action_gym)

        eps_time += 1 
        t_updates += 1
        total_reward += reward
          
        if training_mode: 
            agent.memory.save_eps(state.tolist(), action, reward, float(done), next_state.tolist()) 
            
        state = next_state
                
        if render:
            env.render()     
        
        if training_mode:
            if n_update is not None and t_updates == n_update:
                agent.update_ppo()
                t_updates = 0

                if params_dynamic:
                    params = params - params_subtract
                    params = params if params > params_min else params_min
        
        if done: 
            break                
    
    if training_mode:
        if n_update is None:
            agent.update_ppo()
            t_updates = 0

            if params_dynamic:
                params = params - params_subtract
                params = params if params > params_min else params_min
                
    return total_reward, eps_time, t_updates, params
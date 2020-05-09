import numpy as np
import matplotlib.pyplot as plt
from utils.math_function import prepo_full

def plot(datas):
    print('----------')

    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()

    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))

def run_an_episode(env, agent, render, training_mode, t_updates, n_update, params, params_max, params_min, params_subtract, params_dynamic):
    state       = np.zeros((1, 160, 160))
    next_state  = np.zeros((1, 160, 160))
    ############################################
    obs     = env.reset()
    obs     = np.array(prepo_full(obs)).reshape(1, 160, 160)
    state   = obs

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
        next_obs                    = np.array(prepo_full(next_obs)).reshape(1, 160, 160)
        next_state                  = next_obs - obs         

        eps_time += 1 
        t_updates += 1
        total_reward += reward
          
        if training_mode:  
            agent.memory.save_eps(state.tolist(), action, reward, float(done), next_state.tolist()) 
            
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

def run_discrete(agent, env, n_episode, reward_threshold, save_weights = False, n_plot_batch = 100, render = True, training_mode = True, n_update = 128, n_saved = 10,
        params_max = 1.0, params_min = 0.2, params_subtract = 0.00001, params_dynamic = True):
  
    params = params_max
    #############################################     

    rewards = []   
    batch_rewards = []
    batch_solved_reward = []

    times = []
    batch_times = []

    t_updates = 0
    print('Running the training!!')

    for i_episode in range(1, n_episode + 1):
        total_reward, time, t_updates, params = run_an_episode(env, agent, render, training_mode, t_updates, n_update, params, params_max, params_min, params_subtract, params_dynamic)
        print('Episode {} \t t_reward: {} \t time: {} \t '.format(i_episode, int(total_reward), time))
        batch_rewards.append(int(total_reward))
        batch_times.append(time)        

        if save_weights:
            if i_episode % n_saved == 0:
                agent.save_weights() 
                print('weights saved')

        if reward_threshold:
            if len(batch_solved_reward) == 100:            
                if np.mean(batch_solved_reward) >= reward_threshold :              
                    for reward in batch_rewards:
                        rewards.append(reward)

                    for time in batch_times:
                        times.append(time)                    

                    print('You solved task after {} episode'.format(len(rewards)))
                    break

                else:
                    del batch_solved_reward[0]
                    batch_solved_reward.append(total_reward)

            else:
                batch_solved_reward.append(total_reward)

        if i_episode % n_plot_batch == 0 and i_episode != 0:
            # Plot the reward, times for every n_plot_batch
            plot(batch_rewards)
            plot(batch_times)

            for reward in batch_rewards:
                rewards.append(reward)

            for time in batch_times:
                times.append(time)

            batch_rewards = []
            batch_times = []

            print('========== Cummulative ==========')
            # Plot the reward, times for every episode
            plot(rewards)
            plot(times)

    print('========== Final ==========')
     # Plot the reward, times for every episode
    plot(rewards)
    plot(times)
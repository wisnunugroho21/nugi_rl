import numpy as np
import matplotlib.pyplot as plt
import requests

from memory.on_policy_impala_memory import OnMemoryImpala

def get_action(state):
    data = {
        'state': state
    }

    r = requests.post(url = 'http://localhost:8010/act', json = data)
    data = r.json()
    return data['action'], data['worker_action_datas']

def send_trajectory(states, actions, rewards, dones, next_states, worker_action_datas):
    data = {
        'states'                : states,
        'actions'               : actions,
        'rewards'               : rewards,        
        'dones'                 : dones,
        'next_states'           : next_states,
        'worker_action_datas'   : worker_action_datas
    }

    r = requests.post(url = 'http://localhost:5000/trajectory', json = data)
    data = r.json()

    return data['success']

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

def run_an_episode(memory, env, render, training_mode, max_action):
    ############################################
    state = env.reset()    
    done = False
    total_reward = 0
    eps_time = 0
    ############################################ 
    for _ in range(1, 5000): 
        action, worker_action_datas = get_action(state.tolist())

        if np.isscalar(action):
            action = np.array([action])
        action_env = np.clip(action, -1, 1) * max_action
        
        next_state, reward, done, _ = env.step(action_env)

        eps_time += 1
        total_reward += reward
          
        if training_mode: 
            memory.save_eps(state.tolist(), action, reward, float(done), next_state.tolist(), worker_action_datas)
            
        state = next_state
                
        if render:
            env.render()
        
        if done: 
            send_trajectory(*memory.get_all_items())
            memory.clear_memory()

            break                

    return total_reward, eps_time

def run(env, n_episode, reward_threshold = None, n_plot_batch = 100, render = True, training_mode = True, max_action = 1.0):
    #############################################
    rewards = []   
    batch_rewards = []
    batch_solved_reward = []

    times = []
    batch_times = []

    #############################################

    memory = OnMemoryImpala()

    #############################################

    for i_episode in range(1, n_episode + 1):
        total_reward, time = run_an_episode(memory, env, render, training_mode, max_action)
        print('Episode {} \t t_reward: {} \t time: {} \t '.format(i_episode, int(total_reward), time))
        batch_rewards.append(int(total_reward))
        batch_times.append(time)

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
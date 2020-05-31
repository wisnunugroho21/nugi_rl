import numpy as np
import matplotlib.pyplot as plt

import requests
import redis

from time import sleep

from utils.redis_utils import toRedis, fromRedis

r = redis.Redis()
keys = ['actor_w1', 'actor_w2', 'actor_w3', 'actor_w4', 'actor_w5', 'actor_w6']

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

def run_an_episode(env, agent, render, training_mode, t_updates, n_update, params, params_max, params_min, params_subtract, params_dynamic, max_action):
    ############################################
    state = env.reset()    
    done = False
    total_reward = 0
    eps_time = 0
    ############################################    
    agent.set_params(params) 
    ############################################
    for _ in range(1, 5000): 
        action = agent.act(state)        
        if np.isscalar(action):
            action = np.array([action])
        action_env = np.clip(action, -1, 1) * max_action
        
        next_state, reward, done, _ = env.step(action_env)

        eps_time += 1 
        t_updates += 1
        total_reward += reward
          
        if training_mode: 
            agent.save_eps(state.tolist(), reward, action.tolist(), float(done), next_state.tolist())
            
        state = next_state
                
        if render:
            env.render()     
        
        if training_mode:
            if t_updates == n_update:
                states, actions, rewards, dones, next_states = agent.get_eps()
                
                toRedis(r, np.array(states), 'states')
                toRedis(r, np.array(rewards), 'rewards')
                toRedis(r, np.array(actions), 'actions')
                toRedis(r, np.array(dones), 'dones')
                toRedis(r, np.array(next_states), 'next_states')
                r.set('is_new_params', 1)
                #print(r.get('is_new_params').decode('utf-8'))

                actor_w = []
                #print(r.get('is_optimizing').decode('utf-8'))
                while int(r.get('is_optimizing').decode('utf-8')) == 1:
                    pass

                actor_w1 = fromRedis(r, 'actor_w1')
                actor_w2 = fromRedis(r, 'actor_w2')
                actor_w3 = fromRedis(r, 'actor_w3')
                actor_w4 = fromRedis(r, 'actor_w4')
                actor_w5 = fromRedis(r, 'actor_w5')
                actor_w6 = fromRedis(r, 'actor_w6')

                actor_w.append(actor_w1.tolist())
                actor_w.append(actor_w2.tolist())
                actor_w.append(actor_w3.tolist())
                actor_w.append(actor_w4.tolist())
                actor_w.append(actor_w5.tolist())
                actor_w.append(actor_w6.tolist())
                
                agent.set_weights(actor_w)

                t_updates = 0

                if params_dynamic:
                    params = params - params_subtract
                    params = params if params > params_min else params_min
        
        if done: 
            break                

    return total_reward, eps_time, t_updates, params

def run(agent, env, n_episode, reward_threshold, save_weights = False, n_plot_batch = 100, render = True, training_mode = True, n_update = 1024, n_saved = 10,
        params_max = 1.0, params_min = 0.2, params_subtract = 0.0001, params_dynamic = True, max_action = 1.0):
    params = params_max
    #############################################     

    rewards = []   
    batch_rewards = []
    batch_solved_reward = []

    times = []
    batch_times = []

    t_updates = 0

    actor_w = []
    finishUpdating = False
    while not finishUpdating:
        actor_w1 = fromRedis(r, 'actor_w1')
        actor_w2 = fromRedis(r, 'actor_w2')
        actor_w3 = fromRedis(r, 'actor_w3')
        actor_w4 = fromRedis(r, 'actor_w4')
        actor_w5 = fromRedis(r, 'actor_w5')
        actor_w6 = fromRedis(r, 'actor_w6')

        if actor_w1 is not None and actor_w2 is not None and actor_w3 is not None and actor_w4 is not None and actor_w5 is not None and actor_w6 is not None:
            actor_w.append(actor_w1.tolist())
            actor_w.append(actor_w2.tolist())
            actor_w.append(actor_w3.tolist())
            actor_w.append(actor_w4.tolist())
            actor_w.append(actor_w5.tolist())
            actor_w.append(actor_w6.tolist())

            finishUpdating = True
    
    agent.set_weights(actor_w)
    #[r.delete(key) for key in keys]

    for i_episode in range(1, n_episode + 1):
        total_reward, time, t_updates, params = run_an_episode(env, agent, render, training_mode, t_updates, n_update, params, params_max, params_min, params_subtract, params_dynamic, max_action)
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
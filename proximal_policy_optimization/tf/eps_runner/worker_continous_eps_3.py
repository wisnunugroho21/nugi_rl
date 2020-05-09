from utils.mongodb_utils import Observation, Weight
import numpy as np
import matplotlib.pyplot as plt
from mongoengine import *
import bson

connect()

def set_actor_weights(agent):
    actor_w = []
    actor_w.append(np.ones((64, 24)))
    actor_w.append(np.ones((64,)))
    actor_w.append(np.ones((64, 64)))
    actor_w.append(np.ones((64,)))
    actor_w.append(np.ones((64, 64)))
    actor_w.append(np.ones((64,)))
    actor_w.append(np.ones((4, 64)))
    actor_w.append(np.ones((4,)))

    if Weight.objects.count() > 0:
        for w in Weight.objects:
            aw = actor_w[w.dim1]

            if len(aw.shape) == 1:
                aw[w.dim2,] = w.weight
            elif len(aw.shape) == 2:
                aw[w.dim2, w.dim3] = w.weight

        agent.set_weights(actor_w)

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
            worker_logprobs = agent.log_action(state.tolist(), action.tolist())
            agent.save_eps(state.tolist(), action.tolist(), reward, float(done), next_state.tolist(), worker_logprobs.tolist())
            
        state = next_state
                
        if render:
            env.render()        
        
        if done: 
            if training_mode:
                agent.convert_next_states_to_next_next_states()
                
                states, actions, rewards, dones, next_states, logprobs, next_next_states = agent.get_eps()
                for s, a, r, d, ns, l, nns in zip(states, actions, rewards, dones, next_states, logprobs, next_next_states):
                    obs = Observation(states = s, actions = a, rewards = r, dones = d, next_states = ns, logprobs = l, next_next_states = nns)
                    obs.save()                

                set_actor_weights(agent)
                agent.clearMemory()

                t_updates = 0
                if params_dynamic:
                    params = params - params_subtract
                    params = params if params > params_min else params_min

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

    set_actor_weights(agent)

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
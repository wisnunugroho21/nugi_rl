import numpy as np
import matplotlib.pyplot as plt

from utils.math_function import plot

class StandardExecutor():
    def __init__(self, agent, env, n_episode, Runner, reward_threshold, save_weights = False, n_plot_batch = 100, render = True, training_mode = True, n_update = 1024, n_aux_update = 10, 
        n_saved = 10, params_max = 1.0, params_min = 0.2, params_subtract = 0.0001, params_dynamic = True, max_action = 1.0):

        self.agent = agent        
        self.runner = Runner(env, self.agent, render, training_mode, n_update, n_aux_update, params_max, params_min, params_subtract, params_dynamic)

        self.params_max = params_max
        self.n_episode = n_episode
        self.save_weights = save_weights
        self.n_saved = n_saved
        self.reward_threshold = reward_threshold
        self.n_plot_batch = n_plot_batch
        self.max_action = max_action
        
    def execute_discrete(self):        
        rewards = []   
        batch_rewards = []
        batch_solved_reward = []

        times = []
        batch_times = []
        print('Running the training!!')

        for i_episode in range(1, self.n_episode + 1):
            total_reward, time = self.runner.run_discrete_episode()

            print('Episode {} \t avg reward: {} \t time: {} \t '.format(i_episode, round(total_reward, 2), time))
            batch_rewards.append(int(total_reward))
            batch_times.append(time)        

            if self.save_weights:
                if i_episode % self.n_saved == 0:
                    self.agent.save_weights() 
                    print('weights saved')

            if self.reward_threshold:
                if len(batch_solved_reward) == 100:            
                    if np.mean(batch_solved_reward) >= self.reward_threshold :              
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

            if i_episode % self.n_plot_batch == 0 and i_episode != 0:
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

    def execute_continous(self):
        rewards = []   
        batch_rewards = []
        batch_solved_reward = []

        times = []
        batch_times = []
        print('Running the training!!')

        for i_episode in range(1, self.n_episode + 1): 
            total_reward, time = self.runner.run_continous_episode()

            print('Episode {} \t t_reward: {} \t time: {} \t '.format(i_episode, int(total_reward), time))
            batch_rewards.append(int(total_reward))
            batch_times.append(time)        

            if self.save_weights:
                if i_episode % self.n_saved == 0:
                    self.agent.save_weights() 
                    print('weights saved')

            if self.reward_threshold:
                if len(batch_solved_reward) == 100:            
                    if np.mean(batch_solved_reward) >= self.reward_threshold :              
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

            if i_episode % self.n_plot_batch == 0 and i_episode != 0:
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
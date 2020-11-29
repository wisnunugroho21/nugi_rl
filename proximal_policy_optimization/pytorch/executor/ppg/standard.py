import numpy as np
import matplotlib.pyplot as plt

from utils.math_function import plot
from torch.utils.tensorboard import SummaryWriter

class StandardExecutor():
    def __init__(self, agent, env, n_episode, Runner, reward_threshold, save_weights = False, n_plot_batch = 100, render = True, training_mode = True, n_update = 1024, n_aux_update = 10, 
        n_saved = 10, params_max = 1.0, params_min = 0.2, params_subtract = 0.0001, params_dynamic = True, max_action = 1.0):

        self.agent = agent        
        self.runner = Runner(env, self.agent, render, training_mode, n_update, n_aux_update, params_max, params_min, params_subtract, params_dynamic, max_action)

        self.params_max = params_max
        self.n_episode = n_episode
        self.save_weights = save_weights
        self.n_saved = n_saved
        self.reward_threshold = reward_threshold
        self.n_plot_batch = n_plot_batch
        self.max_action = max_action

        self.writer = SummaryWriter()
        
    def execute_discrete(self):
        print('Running the training!!')

        for i_episode in range(1, self.n_episode + 1):
            total_reward, time = self.runner.run_discrete_episode()

            print('Episode {} \t avg reward: {} \t time: {} \t '.format(i_episode, round(total_reward, 2), time))            

            if self.save_weights:
                if i_episode % self.n_saved == 0:
                    self.agent.save_weights() 
                    print('weights saved')

            if i_episode % self.n_plot_batch == 0:
                self.writer.add_scalar('Rewards', total_reward, i_episode)
                self.writer.add_scalar('Times', time, i_episode)

    def execute_continous(self):
        print('Running the training!!')

        for i_episode in range(1, self.n_episode + 1): 
            total_reward, time = self.runner.run_continous_episode()

            print('Episode {} \t t_reward: {} \t time: {} \t '.format(i_episode, int(total_reward), time))

            if self.save_weights:
                if i_episode % self.n_saved == 0:
                    self.agent.save_weights() 
                    print('weights saved')

            if i_episode % self.n_plot_batch == 0:
                self.writer.add_scalar('Rewards', total_reward, i_episode)
                self.writer.add_scalar('Times', time, i_episode)
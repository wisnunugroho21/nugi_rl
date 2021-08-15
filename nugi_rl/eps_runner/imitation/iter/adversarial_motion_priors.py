import numpy as np
from copy import deepcopy

class IterRunner():
    def __init__(self, agent, teacher, env, goal, is_save_memory, render, n_update, is_discrete, max_action, writer = None, n_plot_batch = 100, 
        coef_task_reward = 1, coef_imitation_reward = 1):

        self.agent              = agent
        self.env                = env
        self.goal               = goal
        self.teacher            = teacher

        self.render             = render
        self.is_save_memory     = is_save_memory
        self.n_update           = n_update
        self.max_action         = max_action
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch
        self.is_discrete        = is_discrete

        self.coef_task_reward       = coef_task_reward
        self.coef_imitation_reward  = coef_imitation_reward

        self.t_updates          = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0

        self.states             = self.env.reset()

    def run(self):
        for _ in range(self.n_update):
            action  = self.agent.act(self.states, self.goal)
            next_state, task_reward, done, _ = self.env.step(action)

            imitation_reward = self.teacher.teach(self.states, next_state, self.goal)
            reward = self.coef_task_reward * task_reward + self.coef_imitation_reward * imitation_reward
            
            if self.is_save_memory:
                self.agent.memory.save_obs(self.states.tolist(), self.goal, action, reward, float(done), next_state.tolist())
                self.teacher.memory.save_policy_obs(self.states.tolist(), self.goal, next_state.tolist())
                
            self.states         = next_state
            self.eps_time       += 1 
            self.total_reward   += reward
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1
                print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, self.total_reward, self.eps_time))

                if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                    self.writer.add_scalar('Rewards', self.total_reward, self.i_episode)
                    self.writer.add_scalar('Times', self.eps_time, self.i_episode)

                self.states         = self.env.reset()
                self.total_reward   = 0
                self.eps_time       = 0    

        return self.agent.memory.get_ranged_items(-self.n_update)
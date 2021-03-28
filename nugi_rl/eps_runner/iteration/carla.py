import numpy as np
from eps_runner.iteration.iter_runner import IterRunner

class CarlaRunner(IterRunner):
    def __init__(self, agent, env, memory, training_mode, render, n_update, is_discrete, max_action, writer = None, n_plot_batch = 100):
        self.env                = env
        self.agent              = agent
        self.render             = render
        self.training_mode      = training_mode
        self.n_update           = n_update
        self.max_action         = max_action
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch

        self.t_updates          = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0
        
        self.images, self.states    = self.env.reset()
        self.memories               = memory  

    def run(self):
        self.memories.clear_memory()       

        for _ in range(self.n_update):
            action                                  = self.agent.act(self.images, self.states)
            next_image, next_state, reward, done, _ = self.env.step(action)
            
            if self.training_mode:
                self.memories.save_eps(self.images, self.states.tolist(), action, reward, float(done), next_state.tolist(), next_image)
                
            self.images         = next_image
            self.states         = next_state
            self.eps_time       += 1 
            self.total_reward   += reward
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1
                self.__print_result(self.i_episode, self.total_reward, self.eps_time, self.tag)               

                self.images, self.states    = self.env.reset()
                self.total_reward           = 0
                self.eps_time               = 0

        # print('Updating agent..')
        return self.memories
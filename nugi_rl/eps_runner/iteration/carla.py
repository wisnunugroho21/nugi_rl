import numpy as np
from eps_runner.iteration.iter_runner import IterRunner

class CarlaRunner(IterRunner):
    def __init__(self, agent, env, training_mode, render, n_update, is_discrete, max_action, writer = None, n_plot_batch = 100):
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

    def _print_result(self, i_episode, total_reward, eps_time):
        print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, self.total_reward, self.eps_time))
        
        if self.writer is not None:
            self.writer.add_scalar('Rewards', total_reward, i_episode)
            self.writer.add_scalar('Times', eps_time, i_episode)

    def run(self):
        for _ in range(self.n_update):
            action                                  = self.agent.act(self.images, self.states)
            action_gym                              = np.clip(action, -1, 1)
            next_image, next_state, reward, done, _ = self.env.step(action_gym)
            
            if self.training_mode:
                self.agent.save_obs(self.images, self.states.tolist(), action, reward, float(done), next_image, next_state.tolist())
                
            self.images         = next_image
            self.states         = next_state
            self.eps_time       += 1 
            self.total_reward   += reward
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1
                self._print_result(self.i_episode, self.total_reward, self.eps_time)               

                self.images, self.states    = self.env.reset()
                self.total_reward           = 0
                self.eps_time               = 0

        return self.agent.memory.get_ranged_items(-self.n_update)
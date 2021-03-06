from eps_runner.iteration.runner import IterRunner

class CarlaRunner(IterRunner):
    def __init__(self, env, render, training_mode, n_update, is_discrete, memories, agent = None, max_action = 1, writer = None, n_plot_batch = 1):
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
        self.memories               = memories        

    def run(self, agent):
        self.memories.clear_memory()       

        for _ in range(self.n_update):
            action                      = agent.act((self.images, self.states))
            next_data, reward, done, _  = self.env.step(action)
            next_image, next_state      = next_data
            
            if self.training_mode:
                self.memories.save_eps((self.images.tolist(), self.states.tolist()), action, reward, float(done), (next_image.tolist(), next_state.tolist()))
                
            self.images, self.states    = next_image, next_state
            self.eps_time               += 1 
            self.total_reward           += reward
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1
                print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, self.total_reward, self.eps_time))

                if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                    self.writer.add_scalar('Rewards', self.total_reward, self.i_episode)
                    self.writer.add_scalar('Times', self.eps_time, self.i_episode)

                self.images, self.states    = self.env.reset()
                self.total_reward           = 0
                self.eps_time               = 0

        # print('Updating agent..')
        return self.memories
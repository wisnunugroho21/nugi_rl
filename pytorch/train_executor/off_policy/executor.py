from torch.utils.tensorboard import SummaryWriter

class OffExecutor():
    def __init__(self, agent, n_episode, runner, reward_threshold, save_weights = False, n_plot_batch = 100, n_saved = 10, max_action = 1.0, writer = SummaryWriter()):

        self.agent  = agent        
        self.runner = runner

        self.n_episode          = n_episode
        self.save_weights       = save_weights
        self.n_saved            = n_saved
        self.reward_threshold   = reward_threshold
        self.n_plot_batch       = n_plot_batch
        self.max_action         = max_action
        self.writer             = writer

    def execute(self):
        print('Running the training!!')

        for i_episode in range(1, self.n_episode + 1): 
            total_reward, time = self.runner.run_episode()

            print('Episode {} \t t_reward: {} \t time: {} \t '.format(i_episode, round(total_reward, 3), time))

            if self.save_weights:
                if i_episode % self.n_saved == 0:
                    self.agent.save_weights() 
                    print('weights saved')

            if i_episode % self.n_plot_batch == 0:
                self.writer.add_scalar('Rewards', total_reward, i_episode)
                self.writer.add_scalar('Times', time, i_episode)
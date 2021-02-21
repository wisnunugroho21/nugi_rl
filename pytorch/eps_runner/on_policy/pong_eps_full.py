import numpy as np
from utils.math_function import prepo_crop
from eps_runner.on_policy.runner import OnRunner

class PongFullRunner(OnRunner):
    def __init__(self, env, render, training_mode, n_update, is_discrete, memories, agent = None, max_action = 1, writer = None, n_plot_batch = 1):
        super().__init__(env, render, training_mode, n_update, is_discrete, memories, agent, max_action, writer, n_plot_batch)

        self.frame = 4

        obs             = self.env.reset()
        obs             = prepo_crop(obs)
        self.states     = np.zeros((self.frame, 160, 160, 3))
        self.states[-1] = obs

    def run_iteration(self, agent):
        self.memories.clear_memory()

        for _ in range(self.n_update):
            action      = agent.act(self.states)
            action_gym  = action + 1 if action != 0 else 0

            reward      = 0
            done        = False
            next_state  = None
            for i in range(self.frame):
                next_obs, reward_temp, done, _  = self.env.step(action_gym)
                next_obs                        = prepo_crop(next_obs).reshape(1, 160, 160, 3)

                reward      += reward_temp
                next_state  = next_obs if i == 0 else np.concatenate((next_state, next_obs), axis = 0)

                if done:
                    if len(next_state) < self.frame:
                        next_obs      = np.zeros((self.frame - len(next_state), 160, 160, 3))
                        next_state    = np.concatenate((next_state, next_obs), axis = 0)  
                    break 
            
            if self.training_mode:
                self.memories.save_eps(self.states.tolist(), action, reward, float(done), next_state.tolist())
                
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
                
                obs             = self.env.reset()
                obs             = prepo_crop(obs)
                self.states     = np.zeros((self.frame, 160, 160, 3))
                self.states[-1] = obs

                self.total_reward   = 0
                self.eps_time       = 0             
        
        return self.memories
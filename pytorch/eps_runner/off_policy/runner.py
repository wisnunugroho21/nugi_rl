import numpy as np

class OffRunner():
    def __init__(self, env, render, training_mode, n_update, is_discrete, memories, agent = None, max_action = 1):
        self.env                = env
        self.render             = render
        self.training_mode      = training_mode
        self.n_update           = n_update
        self.max_action         = max_action
        self.agent              = agent

        self.t_updates          = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0

        self.states             = self.env.reset()
        self.memories           = memories
        self.is_discrete        = is_discrete

    def run_episode(self):
        ############################################
        state = self.env.reset()    
        done = False
        total_reward = 0
        eps_time = 0
        ############################################
        for _ in range(1, 10000): 
            action = self.agent.act(state)

            if self.is_discrete:
                action = int(action)

            if self.max_action is not None and not self.is_discrete:
                action_gym  =  np.clip(action, -1.0, 1.0) * self.max_action
                next_state, reward, done, _ = self.env.step(action_gym)
            else:
                next_state, reward, done, _ = self.env.step(action)

            eps_time += 1 
            self.t_updates += 1
            total_reward += reward
            
            if self.training_mode:
                self.agent.save_eps(state.tolist(), action.tolist(), reward, float(done), next_state.tolist()) 
                
            state = next_state
                    
            if self.render:
                self.env.render()
            
            if self.training_mode:
                self.agent.update()

            if done: 
                break
                    
        return total_reward, eps_time
import numpy as np

from eps_runner.runner import Runner

class StandardRunner(Runner):
    def __init__(self, env, agent, render, training_mode, n_update, params_max, params_min, params_subtract, params_dynamic, max_action):
        self.env = env
        self.agent = agent
        self.render = render
        self.training_mode = training_mode
        self.n_update = n_update
        self.params_max = params_max
        self.params_min = params_min
        self.params_subtract = params_subtract
        self.params_dynamic = params_dynamic
        self.max_action = max_action

    def run_discrete_episode(self, t_updates, params):
        ############################################
        state = self.env.reset()    
        done = False
        total_reward = 0
        eps_time = 0
        ############################################    
        self.agent.set_params(params) 
        ############################################
        for _ in range(10000): 
            action = int( self.agent.act(state))       
            next_state, reward, done, _ =  self.env.step(action)

            eps_time += 1 
            t_updates += 1
            total_reward += reward
            
            if self.training_mode: 
                 self.agent.memory.save_eps(state.tolist(), action, reward, float(done), next_state.tolist()) 
                
            state = next_state
                    
            if self.render:
                self.env.render()     
            
            if self.training_mode:
                if self.n_update is not None and t_updates == self.n_update:
                    self.agent.update_ppo()
                    t_updates = 0

                    if self.params_dynamic:
                        params = params - self.params_subtract
                        params = params if params > self.params_min else self.params_min
            
            if done: 
                break                
        
        if self.training_mode:
            if self.n_update is None:
                self.agent.update_ppo()
                t_updates = 0

                if self.params_dynamic:
                    params = params - self.params_subtract
                    params = params if params > self.params_min else self.params_min
                    
        return total_reward, eps_time, t_updates, params

    def run_continous_episode(self, t_updates, params, max_action):
        ############################################
        state = self.env.reset()    
        done = False
        total_reward = 0
        eps_time = 0
        ############################################ 
        for _ in range(1, 5000): 
            self.agent.set_params(params)
            action = self.agent.act(state) 

            action_gym = np.clip(action, -1.0, 1.0) * self.max_action
            next_state, reward, done, _ = self.env.step(action_gym)

            eps_time += 1 
            t_updates += 1
            total_reward += reward
            
            if self.training_mode:
                self.agent.memory.save_eps(state.tolist(), action.tolist(), reward, float(done), next_state.tolist()) 
                
            state = next_state
                    
            if self.render:
                self.env.render()     
            
            if self.training_mode and self.n_update is not None and t_updates == self.n_update:
                self.agent.update_ppo()
                t_updates = 0

                if self.params_dynamic:
                    params = params - self.params_subtract
                    params = params if params > self.params_min else self.params_min
            
            if done: 
                break                
        
        if self.training_mode and self.n_update is None:
            self.agent.update_ppo()

            if self.params_dynamic:
                params = params - self.params_subtract
                params = params if params > self.params_min else self.params_min
                    
        return total_reward, eps_time, t_updates, params
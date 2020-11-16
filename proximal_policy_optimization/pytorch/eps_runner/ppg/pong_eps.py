import numpy as np
from utils.math_function import prepro_half_one_dim
from eps_runner.ppg.standard import StandardRunner

class PongRunner(StandardRunner):
    def run_discrete_episode(self):
        ############################################
        obs = self.env.reset()  
        obs = prepro_half_one_dim(obs)  
        state = obs

        done = False
        total_reward = 0
        eps_time = 0
        ############################################    
        self.agent.set_params(self.params) 
        ############################################
        for _ in range(10000): 
            action = int(self.agent.act(state)) 
            action_gym = action + 1 if action != 0 else 0

            next_obs, reward, done, _ = self.env.step(action_gym)
            next_obs = prepro_half_one_dim(next_obs)
            next_state = next_obs - obs

            eps_time += 1 
            self.t_updates += 1
            total_reward += reward
            
            if self.training_mode: 
                self.agent.policy_memory.save_eps(state.tolist(), action, reward, float(done), next_state.tolist()) 
                
            state = next_state   
            obs = next_obs
                    
            if self.render:
                self.env.render()     
            
            if self.training_mode and self.n_update is not None and self.t_updates == self.n_update:
                self.agent.update_ppo()
                self.t_updates = 0
                self.t_aux_updates += 1

                if self.params_dynamic:
                    self.params = self.params - self.params_subtract
                    self.params = self.params if self.params > self.params_min else self.params_min

                if self.t_aux_updates == self.n_aux_update:
                    self.agent.update_aux()
                    self.t_aux_updates = 0
            
            if done: 
                break                
        
        if self.training_mode and self.n_update is None:
            self.agent.update_ppo()
            self.t_aux_updates += 1

            if self.params_dynamic:
                self.params = self.params - self.params_subtract
                self.params = self.params if self.params > self.params_min else self.params_min

            if self.t_aux_updates == self.n_aux_update:
                self.agent.update_aux()
                self.t_aux_updates = 0
                    
        return total_reward, eps_time
import numpy as np
from eps_runner.ppg.standard import StandardRunner

class VectorizedRunner(StandardRunner):
    def run_continous_episode(self):
        ############################################
        state = self.env.reset()    
        done = False
        total_reward = 0
        eps_time = 0
        ############################################           
        self.agent.set_params(self.params) 
        ############################################
        for _ in range(1, 10000): 
            self.agent.set_params(self.params)
            action = self.agent.act(state)

            action_gym = np.clip(action * self.max_action, -self.max_action, self.max_action)
            next_state, reward, done, _ = self.env.step(action_gym)

            eps_time += 1 
            self.t_updates += 1
            total_reward += reward
            
            if self.training_mode:
                self.agent.save_eps(state.tolist(), action.tolist(), reward, float(done), next_state.tolist()) 
                
            state = next_state
                    
            if self.render:
                self.env.render()
            
            if self.training_mode and self.n_update is not None and self.t_updates == self.n_update:
                self.agent.update_ppo()
                self.t_updates = 0
                self.t_aux_updates += 1                

                if self.t_aux_updates == self.n_aux_update:
                    self.agent.update_aux()
                    self.t_aux_updates = 0

                if self.params_dynamic:
                    self.params = self.params - self.params_subtract
                    self.params = self.params if self.params > self.params_min else self.params_min

            if done: 
                break                
        
        if self.training_mode and self.n_update is None:
            self.agent.update_ppo()
            self.t_aux_updates += 1           

            if self.t_aux_updates == self.n_aux_update:
                self.agent.update_aux()
                self.t_aux_updates = 0

            if self.params_dynamic:
                self.params = self.params - self.params_subtract
                self.params = self.params if self.params > self.params_min else self.params_min
                    
        return total_reward, eps_time
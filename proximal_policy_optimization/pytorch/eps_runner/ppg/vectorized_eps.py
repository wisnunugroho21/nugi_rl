import numpy as np
from eps_runner.standard import StandardRunner

from memory.list_memory import ListMemory
from environment.vectorized_env import VectorEnv

class VectorizedRunner(StandardRunner):
    def __init__(self, env, agent, render, training_mode, n_update, n_aux_update, params_max, params_min, params_subtract, params_dynamic, max_action = 1):
        super().__init__(env, agent, render, training_mode, n_update, n_aux_update, params_max, params_min, params_subtract, params_dynamic, max_action)

        self.env        = VectorEnv(env)
        self.memories   = [ListMemory() for _ in range(len(env))]

    def run_discrete_episode(self):
        ############################################
        states          = self.env.reset()    
        done            = False
        total_reward    = 0
        eps_time        = 0
        ############################################           
        self.agent.set_params(self.params)
        ############################################
        for _ in range(self.n_update * self.n_aux_update): 
            self.agent.set_params(self.params)
            actions     = self.agent.act(states)
            datas       = self.env.step(actions)

            rewards     = []
            next_states = []
            for state, action, memory, data in zip(states, actions, self.memories, datas):
                next_state, reward, done, _ = data
                rewards.append(reward)
                next_states.append(next_state)
                
                if self.training_mode:
                    memory.save_eps(state.tolist(), action.tolist(), reward, float(done), next_state.tolist())
            
            eps_time += 1 
            self.t_updates += 1
            total_reward += np.mean(rewards)
                
            states = next_states
                    
            if self.render:
                self.env.render()
            
            if self.training_mode and self.n_update is not None and self.t_updates == self.n_update:
                for memory in self.memories:
                    temp_states, temp_actions, temp_rewards, temp_dones, temp_next_states = memory.get_all_items()
                    self.agent.save_all(temp_states, temp_actions, temp_rewards, temp_dones, temp_next_states)
                    memory.clear_memory()

                self.agent.update_ppo()
                self.t_updates = 0
                self.t_aux_updates += 1                

                if self.t_aux_updates == self.n_aux_update:
                    self.agent.update_aux()
                    self.t_aux_updates = 0

                if self.params_dynamic:
                    self.params = self.params - self.params_subtract
                    self.params = self.params if self.params > self.params_min else self.params_min          
        
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

    def run_continous_episode(self):
        ############################################
        states          = self.env.reset()    
        done            = False
        total_reward    = 0
        eps_time        = 0
        ############################################           
        self.agent.set_params(self.params)
        ############################################
        for _ in range(self.n_update * self.n_aux_update): 
            self.agent.set_params(self.params)
            actions     = self.agent.act(states)

            action_gym  = np.clip(actions * self.max_action, -self.max_action, self.max_action)
            datas       = self.env.step(action_gym)

            rewards     = []
            next_states = []
            for state, action, memory, data in zip(states, actions, self.memories, datas):
                next_state, reward, done, _ = data
                rewards.append(reward)
                next_states.append(next_state)
                
                if self.training_mode:
                    memory.save_eps(state.tolist(), action.tolist(), reward, float(done), next_state.tolist())
            
            eps_time += 1 
            self.t_updates += 1
            total_reward += np.mean(rewards)
                
            states = next_states
                    
            if self.render:
                self.env.render()
            
            if self.training_mode and self.n_update is not None and self.t_updates == self.n_update:
                for memory in self.memories:
                    temp_states, temp_actions, temp_rewards, temp_dones, temp_next_states = memory.get_all_items()
                    self.agent.save_all(temp_states, temp_actions, temp_rewards, temp_dones, temp_next_states)
                    memory.clear_memory()

                self.agent.update_ppo()
                self.t_updates = 0
                self.t_aux_updates += 1                

                if self.t_aux_updates == self.n_aux_update:
                    self.agent.update_aux()
                    self.t_aux_updates = 0

                if self.params_dynamic:
                    self.params = self.params - self.params_subtract
                    self.params = self.params if self.params > self.params_min else self.params_min          
        
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
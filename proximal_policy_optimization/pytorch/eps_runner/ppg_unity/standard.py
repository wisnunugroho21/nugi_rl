import numpy as np

from eps_runner.ppg.runner import Runner
from memory.list_memory import ListMemory
from memory.aux_memory import AuxMemory

class StandardRunner(Runner):
    def __init__(self, env, agent, render, training_mode, n_update, n_aux_update, params_max, params_min, params_subtract, params_dynamic, max_action = 1):
        self.env = env
        self.agent = agent
        self.render = render
        self.training_mode = training_mode
        self.n_update = n_update
        self.n_aux_update = n_aux_update
        self.params_max = params_max
        self.params_min = params_min
        self.params_subtract = params_subtract
        self.params_dynamic = params_dynamic
        self.max_action = max_action

        self.t_updates = 0
        self.t_aux_updates = 0
        self.params = self.params_max

        self.behavior_name      = self.env.behavior_name
        self.tracked_agents     = self.env.tracked_agents

        self.policy_memories    = {}
        self.aux_memories    = {}
        for agent_id in self.tracked_agents:
            self.policy_memories[agent_id] = ListMemory()
            self.aux_memories[agent_id] = AuxMemory()     

    def run_discrete_episode(self):
        self.env.reset()
        ############################################
        total_reward    = 0
        eps_time        = 0
        ############################################        
        self.agent.set_params(self.params)
        ############################################ 
        for _ in range(500):
            decisionSteps, _    = self.env.get_steps(self.behavior_name)
            agent_ids           = decisionSteps.agent_id

            states = {}
            actions = {}
            for agent_id in agent_ids: 
                states[agent_id] = decisionSteps[agent_id].obs
                actions[agent_id] = int(self.agent.act(states[agent_id]))

            if len(actions) > 0:
                self.env.set_actions(self.behavior_name, np.stack([value for key, value in actions.items()]))
            self.env.step()
            decisionSteps, terminalSteps = self.env.get_steps(self.behavior_name)

            rewards = {}
            dones = {}
            next_states = {} 
            for agent_id in decisionSteps.agent_id:
                rewards[agent_id] = decisionSteps[agent_id].reward
                dones[agent_id] = False
                next_states[agent_id] = decisionSteps[agent_id].obs                
            
            for agent_id in terminalSteps.agent_id:
                rewards[agent_id] = terminalSteps[agent_id].reward
                dones[agent_id] = True
                next_states[agent_id] = terminalSteps[agent_id].obs

            eps_time        += 1
            self.t_updates  += 1
            total_reward    += np.mean([value for key, value in rewards.items()])
            
            if self.training_mode:
                for track_agent in self.tracked_agents:
                    if track_agent in states and (track_agent in decisionSteps or track_agent in terminalSteps):
                        self.policy_memories[track_agent].save_eps(states[track_agent], actions[track_agent], rewards[track_agent], dones[track_agent], next_states[track_agent])
                                                
            if self.training_mode and self.n_update is not None and self.t_updates == self.n_update:
                self.t_updates = 0
                self.t_aux_updates += 1
                for track_agent in self.tracked_agents:
                    self.policy_memories[track_agent], self.aux_memories[track_agent] = self.agent.update_ppo(self.policy_memories[track_agent], self.aux_memories[track_agent])

                if self.t_aux_updates == self.n_aux_update:
                    self.t_aux_updates = 0
                    for track_agent in self.tracked_agents:
                        self.aux_memories[track_agent] = self.agent.update_aux(self.aux_memories[track_agent])
        
        if self.training_mode and self.n_update is None:
            self.t_aux_updates += 1
            for track_agent in self.tracked_agents:
                self.policy_memories[track_agent], self.aux_memories[track_agent] = self.agent.update_ppo(self.policy_memories[track_agent], self.aux_memories[track_agent])
            
            if self.t_aux_updates == self.n_aux_update:
                self.t_aux_updates = 0
                for track_agent in self.tracked_agents:
                    self.aux_memories[track_agent] = self.agent.update_aux(self.aux_memories[track_agent])
                    
        return total_reward, eps_time

    def run_continous_episode(self):
        self.env.reset()
        ############################################
        total_reward    = 0
        eps_time        = 0
        ############################################        
        self.agent.set_params(self.params)
        ############################################ 
        for _ in range(500):
            decisionSteps, _    = self.env.get_steps(self.behavior_name)
            agent_ids           = decisionSteps.agent_id

            states = {}
            actions = {}
            action_gym = []
            for agent_id in agent_ids: 
                states[agent_id] = decisionSteps[agent_id].obs
                actions[agent_id] = np.squeeze(self.agent.act(states[agent_id]))
                action_gym.append(actions[agent_id])

            if len(actions) > 0:
                self.env.set_actions(self.behavior_name, np.stack(action_gym))
            self.env.step()
            decisionSteps, terminalSteps = self.env.get_steps(self.behavior_name)

            rewards = {}
            dones = {}
            next_states = {} 
            for agent_id in decisionSteps.agent_id:
                rewards[agent_id] = decisionSteps[agent_id].reward
                dones[agent_id] = False
                next_states[agent_id] = decisionSteps[agent_id].obs                
            
            for agent_id in terminalSteps.agent_id:
                rewards[agent_id] = terminalSteps[agent_id].reward
                dones[agent_id] = True
                next_states[agent_id] = terminalSteps[agent_id].obs

            eps_time        += 1
            self.t_updates  += 1
            total_reward    += np.mean([value for key, value in rewards.items()])
            
            if self.training_mode:
                for track_agent in self.tracked_agents:
                    if track_agent in states and (track_agent in decisionSteps or track_agent in terminalSteps):
                        self.policy_memories[track_agent].save_eps(states[track_agent], actions[track_agent], rewards[track_agent], dones[track_agent], next_states[track_agent])
                                                
            if self.training_mode and self.n_update is not None and self.t_updates == self.n_update:
                self.t_updates = 0
                self.t_aux_updates += 1
                for track_agent in self.tracked_agents:
                    self.policy_memories[track_agent], self.aux_memories[track_agent] = self.agent.update_ppo(self.policy_memories[track_agent], self.aux_memories[track_agent])

                if self.t_aux_updates == self.n_aux_update:
                    self.t_aux_updates = 0
                    for track_agent in self.tracked_agents:
                        self.aux_memories[track_agent] = self.agent.update_aux(self.aux_memories[track_agent])
        
        if self.training_mode and self.n_update is None:
            self.t_aux_updates += 1
            for track_agent in self.tracked_agents:
                self.policy_memories[track_agent], self.aux_memories[track_agent] = self.agent.update_ppo(self.policy_memories[track_agent], self.aux_memories[track_agent])
            
            if self.t_aux_updates == self.n_aux_update:
                self.t_aux_updates = 0
                for track_agent in self.tracked_agents:
                    self.aux_memories[track_agent] = self.agent.update_aux(self.aux_memories[track_agent])
                    
        return total_reward, eps_time
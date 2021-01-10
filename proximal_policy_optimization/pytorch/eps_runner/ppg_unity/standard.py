import numpy as np

from eps_runner.runner import Runner
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

        self.env.reset()
        self.behavior_name      = list(self.env.behavior_specs)[0]
        decision_steps, _       = self.env.get_steps(self.behavior_name)
        self.tracked_agents     = decision_steps.agent_id

        self.policy_memories    = {}
        self.aux_memories       = {}
        for agent_id in self.tracked_agents:
            self.policy_memories[agent_id]  = ListMemory()
            self.aux_memories[agent_id]     = AuxMemory()

    def run_discrete_episode(self):
        self.env.reset()
        decisionSteps, _    = self.env.get_steps(self.behavior_name)

        states = {}
        for agent_id in decisionSteps.agent_id:
            states[agent_id] = decisionSteps[agent_id].obs[0]
        ############################################
        total_reward    = 0
        eps_time        = 0
        ############################################        
        self.agent.set_params(self.params)
        ############################################ 
        for _ in range(self.n_update * self.n_aux_update):
            actions = {}
            for id, state in states.items():
                actions[id]   = int(self.agent.act(state))

            if len(actions) > 0:
                action_gym = np.stack([value for key, value in actions.items()])
                self.env.set_actions(self.behavior_name, action_gym)
                
            self.env.step()
            decisionSteps, terminalSteps = self.env.get_steps(self.behavior_name)

            rewards = {}
            dones = {}
            next_states = {}
            terminated_states = {}

            for agent_id in decisionSteps.agent_id:
                rewards[agent_id]       = decisionSteps[agent_id].reward
                dones[agent_id]         = False
                next_states[agent_id]   = np.squeeze(decisionSteps[agent_id].obs)             
            
            for agent_id in terminalSteps.agent_id:
                rewards[agent_id]           = terminalSteps[agent_id].reward
                dones[agent_id]             = not terminalSteps[agent_id].interrupted
                terminated_states[agent_id] = np.squeeze(terminalSteps[agent_id].obs)

            eps_time        += 1
            self.t_updates  += 1
            total_reward    += np.mean([value for key, value in rewards.items()])
            
            if self.training_mode:
                for track_agent in self.tracked_agents:
                    if track_agent in states and track_agent in next_states:
                        self.policy_memories[track_agent].save_eps(states[track_agent], actions[track_agent], rewards[track_agent], float(dones[track_agent]), next_states[track_agent])

                    elif track_agent in states and track_agent in terminated_states:
                        self.policy_memories[track_agent].save_eps(states[track_agent], actions[track_agent], rewards[track_agent], float(dones[track_agent]), terminated_states[track_agent])

            if self.training_mode and self.n_update is not None and self.t_updates == self.n_update:
                self.t_aux_updates += 1
                self.t_updates = 0
                tempMemory = ListMemory()
                    
                for policy_memory in self.policy_memories.values():
                    tempstates, tempactions, temprewards, tempdones, tempnext_states = policy_memory.get_all_items()
                    tempMemory.save_all(tempstates, tempactions, temprewards, tempdones, tempnext_states)
                    policy_memory.clear_memory()

                tempMemory, self.aux_memories = self.agent.update_ppo(tempMemory, self.aux_memories)
                
                if self.t_aux_updates == self.n_aux_update:
                    self.t_aux_updates = 0
                    self.aux_memories = self.agent.update_aux(self.aux_memories)

                if self.params_dynamic:
                    self.params = self.params - self.params_subtract
                    self.params = self.params if self.params > self.params_min else self.params_min

            states      = next_states
        
        if self.training_mode and self.n_update is None:
            self.t_aux_updates += 1
            tempMemory = ListMemory()
                
            for policy_memory in self.policy_memories.values():
                tempstates, tempactions, temprewards, tempdones, tempnext_states = policy_memory.get_all_items()
                tempMemory.save_all(tempstates, tempactions, temprewards, tempdones, tempnext_states)
                policy_memory.clear_memory()

            tempMemory, self.aux_memories = self.agent.update_ppo(tempMemory, self.aux_memories)
            
            if self.t_aux_updates == self.n_aux_update:
                self.t_aux_updates = 0
                self.aux_memories = self.agent.update_aux(self.aux_memories)

            if self.params_dynamic:
                self.params = self.params - self.params_subtract
                self.params = self.params if self.params > self.params_min else self.params_min
                    
        return total_reward, eps_time

    def run_continous_episode(self):
        self.env.reset()
        decisionSteps, _    = self.env.get_steps(self.behavior_name)

        states = {}
        for agent_id in decisionSteps.agent_id:
            states[agent_id] = decisionSteps[agent_id].obs[0]
        ############################################
        total_reward    = 0
        eps_time        = 0
        ############################################        
        self.agent.set_params(self.params)
        ############################################ 
        for _ in range(self.n_update * self.n_aux_update):
            actions = {}            
            for id, state in states.items():
                actions[id]   = self.agent.act(state)

            if len(actions) > 0:
                action_gym = np.stack([value * self.max_action for key, value in actions.items()])
                self.env.set_actions(self.behavior_name, np.clip(action_gym, -self.max_action, self.max_action))
                
            self.env.step()
            decisionSteps, terminalSteps = self.env.get_steps(self.behavior_name)

            rewards = {}
            dones = {}
            next_states = {}
            terminated_states = {}

            for agent_id in decisionSteps.agent_id:
                rewards[agent_id]       = decisionSteps[agent_id].reward
                dones[agent_id]         = False
                next_states[agent_id]   = np.squeeze(decisionSteps[agent_id].obs[0])             
            
            for agent_id in terminalSteps.agent_id:
                rewards[agent_id]           = terminalSteps[agent_id].reward
                dones[agent_id]             = not terminalSteps[agent_id].interrupted
                terminated_states[agent_id] = np.squeeze(terminalSteps[agent_id].obs[0])

            eps_time        += 1
            self.t_updates  += 1
            total_reward    += np.mean([value for key, value in rewards.items()])
            
            if self.training_mode:
                for track_agent in self.tracked_agents:
                    if track_agent in states and track_agent in next_states:
                        self.policy_memories[track_agent].save_eps(states[track_agent], actions[track_agent], rewards[track_agent], float(dones[track_agent]), next_states[track_agent])

                    elif track_agent in states and track_agent in terminated_states:
                        self.policy_memories[track_agent].save_eps(states[track_agent], actions[track_agent], rewards[track_agent], float(dones[track_agent]), terminated_states[track_agent])

            if self.training_mode and self.n_update is not None and self.t_updates == self.n_update:
                self.t_aux_updates += 1
                self.t_updates = 0

                """ tempMemory = ListMemory()

                for track_agent in self.tracked_agents: 
                    tempstates, tempactions, temprewards, tempdones, tempnext_states = self.policy_memories[track_agent].get_all_items()
                    self.aux_memories[track_agent].save_all(tempstates)
                    tempMemory.save_all(tempstates, tempactions, temprewards, tempdones, tempnext_states)
                    self.policy_memories[track_agent].clear_memory()

                self.agent.update_ppo(tempMemory)
                tempMemory.clear_memory() """

                for track_agent in self.tracked_agents: 
                    self.policy_memories[track_agent], self.aux_memories[track_agent] = self.agent.update_ppo(self.policy_memories[track_agent], self.aux_memories[track_agent], False)
                self.agent.update_model_ppo()
                
                if self.t_aux_updates == self.n_aux_update:
                    self.t_aux_updates = 0

                    """ tempAuxMemory = AuxMemory()
                    for track_agent in self.tracked_agents:
                        tempstates = self.aux_memories[track_agent].get_all_items()
                        tempAuxMemory.save_all(tempstates)
                        self.aux_memories[track_agent].clear_memory()

                    self.agent.update_aux(tempAuxMemory)
                    tempAuxMemory.clear_memory() """

                    for track_agent in self.tracked_agents: 
                        self.aux_memories[track_agent] = self.agent.update_aux(self.aux_memories[track_agent], False)
                    self.agent.update_model_aux()

                if self.params_dynamic:
                    self.params = self.params - self.params_subtract
                    self.params = self.params if self.params > self.params_min else self.params_min

            states      = next_states
        
        if self.training_mode and self.n_update is None:
            self.t_aux_updates += 1

            """ tempMemory = ListMemory()
                
            for track_agent in self.tracked_agents: 
                tempstates, tempactions, temprewards, tempdones, tempnext_states = self.policy_memories[track_agent].get_all_items()
                self.aux_memories[track_agent].save_all(tempstates)
                tempMemory.save_all(tempstates, tempactions, temprewards, tempdones, tempnext_states)
                self.policy_memories[track_agent].clear_memory()

            self.agent.update_ppo(tempMemory)
            tempMemory.clear_memory() """

            for track_agent in self.tracked_agents: 
                self.policy_memories[track_agent], self.aux_memories[track_agent] = self.agent.update_ppo(self.policy_memories[track_agent], self.aux_memories[track_agent], False)
            self.agent.update_model_ppo()
        
            if self.t_aux_updates == self.n_aux_update:
                self.t_aux_updates = 0
                
                """ tempAuxMemory = AuxMemory()
                for track_agent in self.tracked_agents:
                    tempstates = self.aux_memories[track_agent].get_all_items()
                    tempAuxMemory.save_all(tempstates)
                    self.aux_memories[track_agent].clear_memory()

                self.agent.update_aux(tempAuxMemory)
                tempAuxMemory.clear_memory() """

                for track_agent in self.tracked_agents: 
                    self.aux_memories[track_agent] = self.agent.update_aux(self.aux_memories[track_agent], False)
                self.agent.update_model_aux()

            if self.params_dynamic:
                self.params = self.params - self.params_subtract
                self.params = self.params if self.params > self.params_min else self.params_min
                    
        return total_reward, eps_time
import torch

from agent.ppg.agent_standard import AgentContinous, AgentDiscrete
from memory.list_memory import ListMemory
from memory.aux_memory import AuxMemory

from torch.utils.data import Dataset, DataLoader

class AgentMultiDiscrete(AgentDiscrete):
    def __init__(self, Policy_Model, Value_Model, state_dim, action_dim,
                is_training_mode = True, policy_kl_range = 0.0008, policy_params = 20, 
                value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, 
                batch_size = 32, PPO_epochs = 4, Aux_epochs = 4, gamma = 0.99, 
                lam = 0.95, learning_rate = 2.5e-4, folder = 'model', use_gpu = True,
                num_agent = 12):
        
        super(AgentMultiDiscrete, self).__init__(Policy_Model, Value_Model, state_dim, action_dim, 
                is_training_mode, policy_kl_range, policy_params, value_clip, 
                entropy_coef, vf_loss_coef, batch_size, PPO_epochs, Aux_epochs,
                gamma, lam, learning_rate, folder, use_gpu)

        self.num_agent = num_agent

        self.policy_memories    = [ListMemory() for _ in range(self.num_agent)]
        self.aux_memories       = [AuxMemory() for _ in range(self.num_agent)]

    def save_eps(self, states, actions, rewards, dones, next_states):
        for i in range(self.num_agent):
            self.policy_memories[i].save_eps(states[i], actions[i], rewards[i], dones[i], next_states[i])

    # Update the model
    def update_ppo(self, memories = None):        
        if memories is None:
            memories = self.policy_memories

        for i in range(self.num_agent):
            dataloader = DataLoader(memories[i], self.batch_size, shuffle = False)

            # Optimize policy for K epochs:
            for _ in range(self.PPO_epochs):       
                for states, actions, rewards, dones, next_states in dataloader: 
                    self.training_ppo(states.float().to(self.device), actions.float().to(self.device), \
                        rewards.float().to(self.device), dones.float().to(self.device), next_states.float().to(self.device))

            # Clear the memory
            states, _, _, _, _ = self.policy_memories[i].get_all_items()
            self.aux_memories[i].save_all(states)
            self.policy_memories[i].clear_memory()

            # Copy new weights into old policy:
            self.policy_old.load_state_dict(self.policy.state_dict())
            self.value_old.load_state_dict(self.value.state_dict())

    def update_aux(self, memory = None):        
        if memory is None:
            memories = self.aux_memories

        for i in range(self.num_agent):
            dataloader  = DataLoader(memories[i], self.batch_size, shuffle = False)

            # Optimize policy for K epochs:
            for _ in range(self.Aux_epochs):       
                for states in dataloader:
                    self.training_aux(states.float().to(self.device))

            # Clear the memory
            self.aux_memory[i].clear_memory()

            # Copy new weights into old policy:
            self.policy_old.load_state_dict(self.policy.state_dict())

class AgentMultiContinous(AgentContinous):
    def __init__(self, Policy_Model, Value_Model, state_dim, action_dim,
                is_training_mode = True, policy_kl_range = 0.0008, policy_params = 20, 
                value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, 
                batch_size = 32, PPO_epochs = 4, Aux_epochs = 4, gamma = 0.99,
                lam = 0.95, learning_rate = 2.5e-4, action_std = 1.0, folder = 'model', use_gpu = True,
                num_agent = 12):
        
        super(AgentMultiContinous, self).__init__(Policy_Model, Value_Model, state_dim, action_dim, 
                is_training_mode, policy_kl_range, policy_params, value_clip, 
                entropy_coef, vf_loss_coef, batch_size, PPO_epochs, Aux_epochs,
                gamma, lam, learning_rate, action_std, folder, use_gpu)

        self.num_agent = num_agent

        self.policy_memories    = [ListMemory() for _ in range(self.num_agent)]
        self.aux_memories       = [AuxMemory() for _ in range(self.num_agent)]

    def save_eps(self, states, actions, rewards, dones, next_states):
        for i in range(self.num_agent):
            if states[i] != None:
                self.policy_memories[i].save_eps(states[i], actions[i], rewards[i], dones[i], next_states[i])

    # Update the model
    def update_ppo(self, memories = None):        
        if memories is None:
            memories = self.policy_memories

        for i in range(self.num_agent):
            dataloader = DataLoader(memories[i], self.batch_size, shuffle = False)

            # Optimize policy for K epochs:
            for _ in range(self.PPO_epochs):       
                for states, actions, rewards, dones, next_states in dataloader: 
                    self.training_ppo(states.float().to(self.device), actions.float().to(self.device), \
                        rewards.float().to(self.device), dones.float().to(self.device), next_states.float().to(self.device))

            # Clear the memory
            states, _, _, _, _ = self.policy_memories[i].get_all_items()
            self.aux_memories[i].save_all(states)
            self.policy_memories[i].clear_memory()

            # Copy new weights into old policy:
            self.policy_old.load_state_dict(self.policy.state_dict())
            self.value_old.load_state_dict(self.value.state_dict())

    def update_aux(self, memory = None):        
        if memory is None:
            memories = self.aux_memories

        for i in range(self.num_agent):
            dataloader  = DataLoader(memories[i], self.batch_size, shuffle = False)

            # Optimize policy for K epochs:
            for _ in range(self.Aux_epochs):       
                for states in dataloader:
                    self.training_aux(states.float().to(self.device))

            # Clear the memory
            self.aux_memory[i].clear_memory()

            # Copy new weights into old policy:
            self.policy_old.load_state_dict(self.policy.state_dict())
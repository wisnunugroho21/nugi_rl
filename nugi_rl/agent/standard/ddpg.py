import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from copy import deepcopy

from helpers.pytorch_utils import set_device, copy_parameters, to_list

class AgentDDPG():
    def __init__(self, soft_q, policy, state_dim, action_dim, q_loss, policy_loss, memory, 
        soft_q_optimizer, policy_optimizer, is_training_mode = True, batch_size = 32, epochs = 1, 
        soft_tau = 0.95, folder = 'model', device = torch.device('cuda:0'), 
        target_policy = None, target_soft_q = None):

        self.batch_size         = batch_size
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.state_dim          = state_dim
        self.folder             = folder
        self.epochs             = epochs
        self.soft_tau           = soft_tau

        self.policy             = policy
        self.soft_q             = soft_q

        self.target_policy      = target_policy
        self.target_soft_q      = target_soft_q
                
        self.qLoss              = q_loss
        self.policyLoss         = policy_loss

        self.agent_memory             = memory
        self.device             = device
        self.q_update           = 1
        
        self.soft_q_optimizer   = soft_q_optimizer
        self.policy_optimizer   = policy_optimizer

        if self.target_policy is None:
            self.target_policy  = deepcopy(self.policy)

        if self.target_soft_q is None:
            self.target_soft_q  = deepcopy(self.soft_q)

    @property
    def memory(self):
        return self.agent_memory

    def _training_q(self, states, actions, rewards, dones, next_states):
        self.soft_q_optimizer.zero_grad()
        with torch.cuda.amp.autocast():            
            predicted_next_actions      = self.target_policy(next_states, True)
            target_next_q               = self.target_soft_q(next_states, predicted_next_actions, True)

            predicted_q_value           = self.soft_q(states, actions)

            loss    = self.qLoss.compute_loss(predicted_q_value, target_next_q, rewards, dones)
        
        self.soft_q_scaler.scale(loss).backward()
        self.soft_q_scaler.step(self.soft_q_optimizer)
        self.soft_q_scaler.update()

    def _training_policy(self, states):
        self.policy_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            predicted_actions   = self.policy(states)
            predicted_q_value   = self.soft_q(states, predicted_actions)

            loss    = self.policyLoss.compute_loss(predicted_q_value)

        self.policy_scaler.scale(loss).backward()
        self.policy_scaler.step(self.policy_optimizer)
        self.policy_scaler.update()

    def _update_offpolicy(self):
        if len(self.agent_memory) > self.batch_size:
            for _ in range(self.epochs):
                indices     = torch.randperm(len(self.agent_memory))[:self.batch_size]
                indices     = len(self.agent_memory) - indices - 1

                dataloader  = DataLoader(self.agent_memory, self.batch_size, sampler = SubsetRandomSampler(indices), num_workers = 8)                
                for states, actions, rewards, dones, next_states in dataloader:
                    self._training_q(states.to(self.device), actions.to(self.device), rewards.to(self.device), dones.to(self.device), next_states.to(self.device))
                    self._training_policy(states.to(self.device))

                    self.target_soft_q = copy_parameters(self.soft_q, self.target_soft_q, self.soft_tau)
                    self.target_policy = copy_parameters(self.policy, self.target_policy, self.soft_tau)

    def update(self):
        if len(self.agent_memory) > self.batch_size:
            self._update_offpolicy()
        
    def act(self, state):
        state   = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action  = self.policy(state)
                      
        return to_list(action.squeeze(), self.use_gpu)

    def save_obs(self, state, action, reward, done, next_state):
        self.agent_memory.save_obs(state, action, reward, done, next_state)
        
    def save_weights(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'soft_q_state_dict': self.soft_q.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'soft_q_optimizer_state_dict': self.soft_q_optimizer.state_dict(),
            'policy_scaler_state_dict': self.policy_scaler.state_dict(),
            'value_scaler_state_dict': self.value_scaler.state_dict(),
            'soft_q_scaler_state_dict': self.soft_q_scaler.state_dict(),
        }, self.folder + '/cql.tar')
        
    def load_weights(self, device = None):
        if device == None:
            device = self.device

        model_checkpoint = torch.load(self.folder + '/cql.tar', map_location = device)
        
        self.policy.load_state_dict(model_checkpoint['policy_state_dict']) 
        self.soft_q.load_state_dict(model_checkpoint['soft_q_state_dict'])
        self.policy_optimizer.load_state_dict(model_checkpoint['policy_optimizer_state_dict'])
        self.soft_q_optimizer.load_state_dict(model_checkpoint['soft_q_optimizer_state_dict'])
        self.policy_scaler.load_state_dict(model_checkpoint['policy_scaler_state_dict']) 
        self.soft_q_scaler.load_state_dict(model_checkpoint['soft_q_scaler_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.value.train()
            print('Model is training...')

        else:
            self.policy.eval()
            self.value.eval()
            print('Model is evaluating...')
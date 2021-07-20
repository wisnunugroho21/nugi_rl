import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from copy import deepcopy

from helpers.pytorch_utils import set_device, copy_parameters, to_list

class AgentCql():
    def __init__(self, soft_q1, soft_q2, policy, state_dim, action_dim, q_loss, policy_loss, memory, 
        soft_q_optimizer, policy_optimizer, is_training_mode = True, batch_size = 32, epochs = 1, 
        soft_tau = 0.95, folder = 'model', use_gpu = True):

        self.batch_size         = batch_size
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.state_dim          = state_dim
        self.folder             = folder
        self.use_gpu            = use_gpu
        self.epochs             = epochs
        self.soft_tau           = soft_tau

        self.policy             = policy
        self.soft_q1            = soft_q1
        self.soft_q2            = soft_q2

        self.target_policy      = deepcopy(self.policy)
        self.target_soft_q1     = deepcopy(self.soft_q1)
        self.target_soft_q2     = deepcopy(self.soft_q2)
                
        self.qLoss              = q_loss
        self.policyLoss         = policy_loss

        self.memory             = memory
        self.device             = set_device(self.use_gpu)
        self.q_update           = 1
        
        self.soft_q_optimizer   = soft_q_optimizer
        self.policy_optimizer   = policy_optimizer

        self.soft_q_scaler      = torch.cuda.amp.GradScaler()
        self.policy_scaler      = torch.cuda.amp.GradScaler()

    def _training_q(self, states, actions, rewards, dones, next_states):
        self.soft_q_optimizer.zero_grad()
        with torch.cuda.amp.autocast():            
            predicted_next_actions      = self.target_policy(next_states, True)
            target_next_q1              = self.target_soft_q1(next_states, predicted_next_actions, True)
            target_next_q2              = self.target_soft_q2(next_states, predicted_next_actions, True)

            predicted_actions           = self.policy(states, True)
            naive_predicted_q_value1    = self.soft_q1(states, predicted_actions)
            naive_predicted_q_value2    = self.soft_q2(states, predicted_actions)

            predicted_q_value1          = self.soft_q1(states, actions)
            predicted_q_value2          = self.soft_q2(states, actions)

            loss    = self.qLoss.compute_loss(predicted_q_value1, naive_predicted_q_value1, predicted_q_value2, naive_predicted_q_value2, target_next_q1, target_next_q2, rewards, dones)
        
        self.soft_q_scaler.scale(loss).backward()
        self.soft_q_scaler.step(self.soft_q_optimizer)
        self.soft_q_scaler.update()

    def _training_policy(self, states):
        self.policy_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            predicted_actions   = self.policy(states)

            predicted_q_value1  = self.soft_q1(states, predicted_actions)
            # predicted_q_value2  = self.soft_q2(states, predicted_actions)

            loss    = self.policyLoss.compute_loss(predicted_q_value1)

        self.policy_scaler.scale(loss).backward()
        self.policy_scaler.step(self.policy_optimizer)
        self.policy_scaler.update()

    def _update_offpolicy(self):
        if len(self.memory) > self.batch_size:
            for _ in range(self.epochs):
                indices     = torch.randperm(len(self.memory))[:self.batch_size]
                indices     = len(self.memory) - indices - 1

                dataloader  = DataLoader(self.memory, self.batch_size, sampler = SubsetRandomSampler(indices), num_workers = 8)
                for states, actions, rewards, dones, next_states in dataloader:
                    if self.q_update == 1:
                        self._training_q(states.to(self.device), actions.to(self.device), rewards.to(self.device), dones.to(self.device), next_states.to(self.device))
                        
                        self.target_soft_q1 = copy_parameters(self.soft_q1, self.target_soft_q1, self.soft_tau)
                        self.target_soft_q2 = copy_parameters(self.soft_q2, self.target_soft_q2, self.soft_tau)

                        self.q_update = 2

                    else:
                        self._training_q(states.to(self.device), actions.to(self.device), rewards.to(self.device), dones.to(self.device), next_states.to(self.device))
                        self._training_policy(states.to(self.device))

                        self.target_soft_q1 = copy_parameters(self.soft_q1, self.target_soft_q1, self.soft_tau)
                        self.target_soft_q2 = copy_parameters(self.soft_q2, self.target_soft_q2, self.soft_tau)
                        self.target_policy  = copy_parameters(self.policy, self.target_policy, self.soft_tau)
                        
                        self.q_update = 1

    def save_memory(self, policy_memory):
        states, actions, rewards, dones, next_states = policy_memory.get_all_items()
        self.memory.save_all(states, actions, rewards, dones, next_states)
        
    def act(self, state):
        state   = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action  = self.policy(state)
                      
        return to_list(action.squeeze(), self.use_gpu)

    def update(self):
        self._update_offpolicy()
        
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
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        self.soft_q.load_state_dict(model_checkpoint['soft_q_state_dict'])
        self.policy_optimizer.load_state_dict(model_checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(model_checkpoint['value_optimizer_state_dict'])
        self.soft_q_optimizer.load_state_dict(model_checkpoint['soft_q_optimizer_state_dict'])
        self.policy_scaler.load_state_dict(model_checkpoint['policy_scaler_state_dict'])        
        self.value_scaler.load_state_dict(model_checkpoint['value_scaler_state_dict'])
        self.soft_q_scaler.load_state_dict(model_checkpoint['soft_q_scaler_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.value.train()
            print('Model is training...')

        else:
            self.policy.eval()
            self.value.eval()
            print('Model is evaluating...')
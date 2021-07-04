import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from copy import deepcopy

from helpers.pytorch_utils import set_device, to_list, copy_parameters

class AgentSAC():
    def __init__(self, soft_q1, soft_q2, policy, value, state_dim, action_dim, distribution, q_loss, policy_loss, value_loss, memory, 
        soft_q_optimizer, policy_optimizer, value_optimizer, is_training_mode = True, batch_size = 32, epochs = 1, 
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
        self.value              = value

        self.target_value       = deepcopy(self.value)         

        self.distribution       = distribution
        self.memory             = memory
        
        self.qLoss              = q_loss
        self.policyLoss         = policy_loss
        self.valueLoss          = value_loss

        self.device             = set_device(self.use_gpu)
        self.q_update           = 1
        
        self.soft_q_optimizer   = soft_q_optimizer
        self.policy_optimizer   = policy_optimizer
        self.value_optimizer    = value_optimizer

        self.soft_q_scaler      = torch.cuda.amp.GradScaler()
        self.policy_scaler      = torch.cuda.amp.GradScaler()
        self.value_scaler       = torch.cuda.amp.GradScaler()

    def _training_q(self, states, actions, rewards, dones, next_states):
        self.soft_q_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            target_values      = self.target_value(next_states, True)

            predicted_q_value1  = self.soft_q1(states, torch.tanh(actions))
            predicted_q_value2  = self.soft_q2(states, torch.tanh(actions))

            loss  = self.qLoss.compute_loss(predicted_q_value1, predicted_q_value2, target_values, rewards, dones)

        self.soft_q_scaler.scale(loss).backward()
        self.soft_q_scaler.step(self.soft_q_optimizer)
        self.soft_q_scaler.update()

    def _training_value(self, states):
        self.value_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            action_datas    = self.policy(states, True)
            actions         = self.distribution.sample(action_datas).detach()

            q_value1        = self.soft_q1(states, torch.tanh(actions), True)
            q_value2        = self.soft_q2(states, torch.tanh(actions), True)

            predicted_value = self.value(states)

            loss    = self.valueLoss.compute_loss(predicted_value, action_datas, actions, q_value1, q_value2)

        self.value_scaler.scale(loss).backward()
        self.value_scaler.step(self.value_optimizer)
        self.value_scaler.update()

    def _training_policy(self, states):
        self.policy_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            action_datas    = self.policy(states)
            actions         = self.distribution.sample(action_datas)

            q_value1        = self.soft_q1(states, torch.tanh(actions))
            q_value2        = self.soft_q2(states, torch.tanh(actions))

            loss = self.policyLoss.compute_loss(action_datas, actions, q_value1, q_value2)

        self.policy_scaler.scale(loss).backward()
        self.policy_scaler.step(self.policy_optimizer)
        self.policy_scaler.update()

    def _update_sac(self):
        if len(self.memory) > self.batch_size:
            for _ in range(self.epochs):
                indices     = torch.randperm(len(self.memory))[:self.batch_size]
                indices[-1] = torch.IntTensor([len(self.memory) - 1])

                dataloader  = DataLoader(self.memory, self.batch_size, sampler = SubsetRandomSampler(indices), num_workers = 8)
                for states, actions, rewards, dones, next_states in dataloader:
                    self._training_value(states.to(self.device))
                    self._training_q(states.to(self.device), actions.to(self.device), rewards.to(self.device), dones.to(self.device), next_states.to(self.device))
                    self._training_policy(states.to(self.device))

                    self.target_value = copy_parameters(self.value, self.target_value, self.soft_tau)

    def update(self):
        self._update_sac()

    def save_memory(self, policy_memory):
        states, actions, rewards, dones, next_states = policy_memory.get_all_items()
        self.memory.save_all(states, actions, rewards, dones, next_states)
        
    def act(self, state):
        state               = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_datas        = self.policy(state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.act_deterministic(action_datas)
              
        return to_list(action.squeeze(), self.use_gpu)

    def save_weights(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'soft_q1_state_dict': self.soft_q1.state_dict(),
            'soft_q2_state_dict': self.soft_q2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'soft_q_optimizer_state_dict': self.soft_q_optimizer.state_dict(),
            'policy_scaler_state_dict': self.policy_scaler.state_dict(),
            'soft_q_scaler_state_dict': self.soft_q_scaler.state_dict(),
        }, self.folder + '/sac.tar')
        
    def load_weights(self, device = None):
        if device == None:
            device = self.device

        model_checkpoint = torch.load(self.folder + '/cql.tar', map_location = device)
        
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])
        self.soft_q1.load_state_dict(model_checkpoint['soft_q1_state_dict'])
        self.soft_q2.load_state_dict(model_checkpoint['soft_q2_state_dict'])
        self.policy_optimizer.load_state_dict(model_checkpoint['policy_optimizer_state_dict'])
        self.soft_q_optimizer.load_state_dict(model_checkpoint['soft_q_optimizer_state_dict'])
        self.policy_scaler.load_state_dict(model_checkpoint['policy_scaler_state_dict'])
        self.soft_q_scaler.load_state_dict(model_checkpoint['soft_q_scaler_state_dict'])
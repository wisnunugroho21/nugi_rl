import torch
from torch.utils.data import DataLoader
import copy

from helpers.pytorch_utils import set_device, to_numpy, copy_parameters

class AgentSAC():
    def __init__(self, soft_q1, soft_q2, policy, state_dim, action_dim, distribution, q_loss, policy_loss, memory, 
        soft_q_optimizer1, soft_q_optimizer2, policy_optimizer, is_training_mode = True, batch_size = 32, epochs = 1, 
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
        self.target_soft_q1     = copy.deepcopy(self.soft_q1)

        self.soft_q2            = soft_q2
        self.target_soft_q2     = copy.deepcopy(self.soft_q2)             

        self.distribution       = distribution
        self.memory             = memory
        
        self.qLoss              = q_loss
        self.policyLoss         = policy_loss

        self.device             = set_device(self.use_gpu)
        self.i_update           = 0
        
        self.soft_q_optimizer1  = soft_q_optimizer1
        self.soft_q_optimizer2  = soft_q_optimizer2
        self.policy_optimizer   = policy_optimizer

        self.soft_q_scaler1     = torch.cuda.amp.GradScaler()
        self.soft_q_scaler2     = torch.cuda.amp.GradScaler()
        self.policy_scaler      = torch.cuda.amp.GradScaler()        

    def _training_q(self, states, actions, rewards, dones, next_states):
        self.soft_q_optimizer1.zero_grad()
        with torch.cuda.amp.autocast():
            action_datas        = self.policy(states, True)
            predicted_actions   = self.distribution.sample(action_datas).detach()

            target_q_value1     = self.target_soft_q1(next_states, predicted_actions, True)
            target_q_value2     = self.target_soft_q2(next_states, predicted_actions, True)

            predicted_q_value1  = self.soft_q1(states, actions)

            loss = self.qLoss.compute_loss(predicted_q_value1, target_q_value1, target_q_value2, action_datas, actions, rewards, dones)

        self.soft_q_scaler1.scale(loss).backward()
        self.soft_q_scaler1.step(self.soft_q_optimizer1)
        self.soft_q_scaler1.update()

        self.soft_q_optimizer2.zero_grad()
        with torch.cuda.amp.autocast():
            action_datas        = self.policy(states, True)
            predicted_actions   = self.distribution.sample(action_datas).detach()

            target_q_value1     = self.target_soft_q1(next_states, predicted_actions, True)
            target_q_value2     = self.target_soft_q2(next_states, predicted_actions, True)

            predicted_q_value2  = self.soft_q2(states, actions)

            loss = self.qLoss.compute_loss(predicted_q_value2, target_q_value1, target_q_value2, action_datas, actions, rewards, dones)

        self.soft_q_scaler2.scale(loss).backward()
        self.soft_q_scaler2.step(self.soft_q_optimizer2)
        self.soft_q_scaler2.update()

    def _training_policy(self, states):
        self.policy_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            action_datas        = self.policy(states)
            actions             = self.distribution.sample(action_datas)

            predicted_q_value1  = self.soft_q1(states, actions)
            predicted_q_value2  = self.soft_q2(states, actions)

            loss = self.policyLoss.compute_loss(action_datas, actions, predicted_q_value1, predicted_q_value2)

        self.policy_scaler.scale(loss).backward()
        self.policy_scaler.step(self.policy_optimizer)
        self.policy_scaler.update()

    def _update_sac(self):
        if len(self.memory) > self.batch_size:
            for _ in range(self.epochs):
                dataloader  = DataLoader(self.memory, self.batch_size, shuffle = True, num_workers = 2)
                states, actions, rewards, dones, next_states = next(iter(dataloader))

                self._training_q(states.float().to(self.device), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), next_states.float().to(self.device))
                self._training_policy(states.float().to(self.device))

            self.target_soft_q1 = copy_parameters(self.soft_q1, self.target_soft_q1, self.soft_tau)
            self.target_soft_q2 = copy_parameters(self.soft_q2, self.target_soft_q2, self.soft_tau)

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
              
        return to_numpy(action, self.use_gpu)    

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
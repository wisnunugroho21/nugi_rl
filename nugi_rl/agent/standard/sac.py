import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from copy import deepcopy

from helpers.pytorch_utils import set_device, to_list, copy_parameters

class AgentSAC():
    def __init__(self, soft_q1, soft_q2, policy, value, distribution, q_loss, policy_loss, value_loss, memory, 
        soft_q_optimizer, policy_optimizer, value_optimizer, is_training_mode = True, batch_size = 32, epochs = 1, 
        soft_tau = 0.95, folder = 'model', device = torch.device('cuda:0'), target_value = None):

        self.batch_size         = batch_size
        self.is_training_mode   = is_training_mode
        self.folder             = folder
        self.epochs             = epochs
        self.soft_tau           = soft_tau

        self.policy             = policy
        self.soft_q1            = soft_q1
        self.soft_q2            = soft_q2
        self.value              = value

        self.target_value       = target_value

        self.distribution       = distribution
        self.agent_memory       = memory
        
        self.qLoss              = q_loss
        self.policyLoss         = policy_loss
        self.valueLoss          = value_loss

        self.device             = device
        self.q_update           = 1
        
        self.soft_q_optimizer   = soft_q_optimizer
        self.policy_optimizer   = policy_optimizer
        self.value_optimizer    = value_optimizer

        if self.target_value is None:
            self.target_value = deepcopy(self.value)

    @property
    def memory(self):
        return self.agent_memory

    def _training_q(self, states, actions, rewards, dones, next_states):
        target_values      = self.target_value(next_states, True)

        predicted_q_value1  = self.soft_q1(states, actions)
        predicted_q_value2  = self.soft_q2(states, actions)

        loss  = self.qLoss.compute_loss(predicted_q_value1, predicted_q_value2, target_values, rewards, dones)

        self.soft_q_optimizer.zero_grad()
        loss.backward()
        self.soft_q_optimizer.step()

    def _training_value(self, states):
        action_datas    = self.policy(states, True)
        actions         = self.distribution.sample(action_datas).detach()

        q_value1        = self.soft_q1(states, actions, True)
        q_value2        = self.soft_q2(states, actions, True)

        predicted_value = self.value(states)

        loss    = self.valueLoss.compute_loss(predicted_value, action_datas, actions, q_value1, q_value2)

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def _training_policy(self, states):
        action_datas    = self.policy(states)
        actions         = self.distribution.sample(action_datas)

        q_value1        = self.soft_q1(states, actions)
        q_value2        = self.soft_q2(states, actions)

        loss = self.policyLoss.compute_loss(action_datas, actions, q_value1, q_value2)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def _update_sac(self):        
        for _ in range(self.epochs):
            indices     = torch.randperm(len(self.agent_memory))[:self.batch_size]
            indices[-1] = torch.IntTensor([len(self.agent_memory) - 1])
            # indices     = torch.arange(-self.batch_size, 0)

            dataloader  = DataLoader(self.agent_memory, self.batch_size, sampler = SubsetRandomSampler(indices))                
            for states, actions, rewards, dones, next_states in dataloader:                
                self._training_q(states.to(self.device), actions.to(self.device), rewards.to(self.device), dones.to(self.device), next_states.to(self.device))
                self._training_value(states.to(self.device))
                self._training_policy(states.to(self.device))

                self.target_value = copy_parameters(self.value, self.target_value, self.soft_tau)

    def update(self):
        if len(self.agent_memory) > self.batch_size:
            self._update_sac()

    def save_memory(self, policy_memory):
        states, actions, rewards, dones, next_states = policy_memory.get_all_items()
        self.agent_memory.save_all(states, actions, rewards, dones, next_states)
        
    def act(self, state):
        state               = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_datas        = self.policy(state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.act_deterministic(action_datas)
              
        return action.squeeze().detach().tolist()

    def logprobs(self, state, action):
        state           = torch.FloatTensor(state).unsqueeze(0).float().to(self.device)
        action          = torch.FloatTensor(action).unsqueeze(0).float().to(self.device)

        action_datas, _ = self.policy(state)
        logprobs        = self.distribution.logprob(action_datas, action)

        return logprobs.squeeze().detach().tolist()

    def save_obs(self, state, action, reward, done, next_state):
        self.agent_memory.save_obs(state, action, reward, done, next_state)

    def save_weights(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'soft_q1_state_dict': self.soft_q1.state_dict(),
            'soft_q2_state_dict': self.soft_q2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'soft_q_optimizer_state_dict': self.soft_q_optimizer.state_dict(),
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
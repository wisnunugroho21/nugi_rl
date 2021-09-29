
from copy import deepcopy
import torch
from torch.utils.data import DataLoader

class AgentVMPO():  
    def __init__(self, policy, value, distribution, alpha_loss, phi_loss, temperature_loss, value_loss,
            policy_memory, policy_optimizer, value_optimizer, policy_epochs = 1, is_training_mode = True, batch_size = 32, folder = 'model', 
            device = torch.device('cuda:0'), old_policy = None, old_value = None):   

        self.batch_size         = batch_size  
        self.policy_epochs      = policy_epochs
        self.is_training_mode   = is_training_mode
        self.folder             = folder

        self.policy             = policy
        self.old_policy         = old_policy
        self.value              = value
        self.old_value          = old_value

        self.distribution       = distribution
        self.policy_memory      = policy_memory
        
        self.alpha_loss         = alpha_loss
        self.phi_loss           = phi_loss
        self.temperature_loss   = temperature_loss
        self.value_loss         = value_loss

        self.policy_optimizer   = policy_optimizer
        self.value_optimizer    = value_optimizer   
        self.device             = device

        self.i_update           = 0

        if self.old_policy is None:
            self.old_policy  = deepcopy(self.policy)

        if self.old_value is None:
            self.old_value  = deepcopy(self.value)

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()
        
    @property
    def memory(self):
        return self.policy_memory

    def _training(self, states, actions, rewards, dones, next_states):
        action_datas, temperature, alpha    = self.policy(states)
        old_action_datas, _, _              = self.old_policy(states, True)       
        values                              = self.value(states)
        old_values                          = self.old_value(states, True)
        next_values                         = self.value(next_states, True) 
        
        loss    = self.phi_loss.compute_loss(action_datas, values, next_values, actions, rewards, dones, temperature) + \
                    self.temperature_loss.compute_loss(values, next_values, rewards, dones, temperature) + \
                    self.alpha_loss.compute_loss(action_datas, old_action_datas, alpha) + \
                    self.value_loss.compute_loss(values, old_values, next_values, rewards, dones) - \
                    0.1 * self.distribution.entropy(action_datas).mean()

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        loss.backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

    def update(self):
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_value.load_state_dict(self.value.state_dict())

        for _ in range(self.policy_epochs):
            dataloader = DataLoader(self.policy_memory, self.batch_size, shuffle = False)
            for states, actions, rewards, dones, next_states in dataloader:
                self._training(states.float().to(self.device), actions.float().to(self.device), rewards.float().to(self.device), dones.float().to(self.device), next_states.float().to(self.device))

        self.policy_memory.clear_memory()

    def act(self, state):
        with torch.inference_mode():
            state               = torch.FloatTensor(state).unsqueeze(0).float().to(self.device)
            action_datas, _, _  = self.old_policy(state)
            
            if self.is_training_mode:
                action = self.distribution.sample(action_datas)
            else:
                action = self.distribution.deterministic(action_datas)

            action = action.squeeze(0).detach().tolist()
              
        return action

    def logprobs(self, state, action):
        with torch.inference_mode():
            state               = torch.FloatTensor(state).unsqueeze(0).float().to(self.device)
            action_datas, _, _  = self.old_policy(state)
            
            logprobs        = self.distribution.logprob(action_datas, action)
            logprobs        = logprobs.squeeze(0).detach().tolist()

        return logprobs

    def save_obs(self, state, action, reward, done, next_state):
        self.policy_memory.save_obs(state, action, reward, done, next_state)

    def save_weights(self, folder = None):
        if folder == None:
            folder = self.folder
            
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, self.folder + '/v_mpo.tar')
        
    def load_weights(self, folder = None, device = None):
        if device == None:
            device = self.device

        if folder == None:
            folder = self.folder

        model_checkpoint = torch.load(self.folder + '/v_mpo.tar', map_location = device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        
        if self.policy_optimizer is not None:
            self.policy_optimizer.load_state_dict(model_checkpoint['policy_optimizer_state_dict'])

        if self.value_optimizer is not None:
            self.value_optimizer.load_state_dict(model_checkpoint['value_optimizer_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.value.train()

        else:
            self.policy.eval()
            self.value.eval()

    def get_weights(self):
        return self.policy.state_dict(), self.value.state_dict()

    def set_weights(self, policy_weights, value_weights):
        self.policy.load_state_dict(policy_weights)
        self.value.load_state_dict(value_weights)
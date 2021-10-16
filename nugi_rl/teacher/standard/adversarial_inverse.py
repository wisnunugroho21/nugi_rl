import torch
from torch.utils.data import DataLoader

class TeacherAdvInv():
    def __init__(self, g_model, h_model, loss_fn, memory, optimizer, epochs = 10, device = torch.device('cuda:0'), is_training_mode = True, batch_size = 32):
        self.g_model            = g_model
        self.h_model            = h_model

        self.memory             = memory
        self.optimizer          = optimizer
        self.loss_fn            = loss_fn

        self.epochs             = epochs
        self.batch_size         = batch_size        
        self.device             = device
        self.is_training_mode   = is_training_mode

    @property
    def memory(self):
        return self.memory

    def _training_rewards(self, expert_states, expert_actions, expert_logprobs, expert_dones, expert_next_states,
        policy_states, policy_actions, policy_logprobs, policy_dones, policy_next_states):

        expert_g_values         = self.g_model(expert_states, expert_actions)
        expert_h_values         = self.h_model(expert_states)
        expert_h_next_values    = self.h_model(expert_next_states)

        policy_g_values         = self.g_model(policy_states, policy_actions)
        policy_h_values         = self.h_model(policy_states)
        policyt_h_next_values   = self.h_model(policy_next_states)

        loss = self.loss_fn.compute_loss(expert_g_values, expert_h_values, expert_h_next_values, expert_logprobs, expert_dones,
            policy_g_values, policy_h_values, policyt_h_next_values, policy_logprobs, policy_dones)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_rewards(self):
        for _ in range(self.epochs):
            dataloader = DataLoader(self.memory, self.batch_size, shuffle = False, num_workers = 8)

            for expert_states, expert_actions, expert_logprobs, expert_dones, expert_next_states, \
                policy_states, policy_actions, policy_logprobs, policy_dones, policy_next_states in dataloader:

                self._training_rewards(expert_states.to(self.device), expert_actions.to(self.device), expert_logprobs.to(self.device), expert_dones.to(self.device), expert_next_states.to(self.device),
                    policy_states.to(self.device), policy_actions.to(self.device), policy_logprobs.to(self.device), policy_dones.to(self.device), policy_next_states.to(self.device))

        self.memory.clear_policy_memory()

    def update(self):
        if len(self.memory) >= self.batch_size:
            self._update_rewards()

    def teach(self, state, action, logprob, done, next_state):
        state       = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action      = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        logprob     = torch.FloatTensor(logprob).unsqueeze(0).to(self.device)
        done        = torch.FloatTensor(done).unsqueeze(0).to(self.device)
        next_state  = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        g_values        = self.g_model(state, action)
        h_values        = self.h_model(state)
        h_next_values   = self.h_model(next_state)
        
        discrimination  = self.loss_fn.compute_descrimination(self, g_values, h_values, h_next_values, logprob, done)
        reward          = discrimination.log() - (1 - discrimination).log()
        
        return reward.squeeze().detach().tolist()

    def save_weights(self, folder = None):
        if folder == None:
            folder = self.folder
            
        torch.save({
            'g_model_state_dict': self.g_model.state_dict(),
            'h_model_state_dict': self.h_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.folder + '/ppg.tar')
        
    def load_weights(self, folder = None, device = None):
        if device == None:
            device = self.device

        if folder == None:
            folder = self.folder

        model_checkpoint = torch.load(self.folder + '/ppg.tar', map_location = device)
        self.g_model.load_state_dict(model_checkpoint['g_model_state_dict'])        
        self.h_model.load_state_dict(model_checkpoint['h_model_state_dict'])
        self.optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])

        if self.is_training_mode:
            self.g_model.train()
            self.h_model.train()

        else:
            self.g_model.eval()
            self.h_model.eval()

    
    
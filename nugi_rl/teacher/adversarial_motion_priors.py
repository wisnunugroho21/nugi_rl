import torch
from torch.utils.data import DataLoader

class TeacherAdvMtnPrio():
    def __init__(self, discrim_model, loss_fn, memory, optimizer, epochs, device, is_training_mode = True, batch_size = 32):
        self.discrim_model      = discrim_model

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

    def _training_rewards(self, expert_states, expert_next_states, policy_states, policy_next_states, goals):
        dis_expert  = self.discrim_model(expert_states, expert_next_states, goals)
        dis_policy  = self.discrim_model(policy_states, policy_next_states, goals)       

        loss = self.loss_fn.compute_loss(dis_expert, dis_policy, policy_states, policy_next_states)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_rewards(self):
        for _ in range(self.epochs):
            dataloader = DataLoader(self.memory, self.batch_size, shuffle = False, num_workers = 8)

            for expert_states, expert_next_states, policy_states, policy_next_states, goals in dataloader:
                self._training_rewards(expert_states.to(self.device), expert_next_states.to(self.device),
                    policy_states.to(self.device), policy_next_states.to(self.device), goals.to(self.device))

        self.memory.clear_policy_memory()

    def update(self):
        if len(self.memory) >= self.batch_size:
            self._update_rewards()

    def teach(self, state, next_state, goal):
        state       = torch.FloatTensor(state).unsqueeze(0).float().to(self.device)
        goal        = torch.FloatTensor(goal).unsqueeze(0).float().to(self.device)
        next_state  = torch.FloatTensor(next_state).unsqueeze(0).float().to(self.device)

        discrimination  = self.discrim_model(state, next_state, goal)
        reward          = torch.max(0, 1 - 0.25 * (discrimination - 1).pow(2))
        
        return reward.squeeze().detach().tolist()

    def save_weights(self, folder = None):
        if folder == None:
            folder = self.folder
            
        torch.save({
            'discrim_model_state_dict': self.discrim_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.folder + '/ppg.tar')
        
    def load_weights(self, folder = None, device = None):
        if device == None:
            device = self.device

        if folder == None:
            folder = self.folder

        model_checkpoint = torch.load(self.folder + '/ppg.tar', map_location = device)
        self.discrim_model.load_state_dict(model_checkpoint['discrim_model_state_dict'])
        self.optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])

        if self.is_training_mode:
            self.discrim_model.train()
        else:
            self.discrim_model.eval()

    
    
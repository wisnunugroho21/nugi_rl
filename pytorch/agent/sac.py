import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.pytorch_utils import set_device, to_numpy, to_tensor

class AgentSAC():
    def __init__(self, Q_Model, Value_Model, Policy_Model, state_dim, action_dim, distribution, q_loss, v_loss, policy_loss, memory, is_training_mode = True, 
        batch_size = 32, epochs = 1, soft_tau = 0.95, learning_rate = 3e-4, folder = 'model', use_gpu = True):

        self.batch_size         = batch_size
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.state_dim          = state_dim
        self.learning_rate      = learning_rate
        self.folder             = folder
        self.use_gpu            = use_gpu
        self.epochs             = epochs
        self.soft_tau           = soft_tau

        self.value              = Value_Model(state_dim, self.use_gpu)
        self.target_value       = Value_Model(state_dim, self.use_gpu)
        self.soft_q1            = Q_Model(state_dim, action_dim, self.use_gpu)
        self.soft_q2            = Q_Model(state_dim, action_dim, self.use_gpu)
        self.policy             = Policy_Model(state_dim, action_dim, self.use_gpu)

        self.distribution       = distribution
        self.memory             = memory
        
        self.qLoss              = q_loss
        self.vLoss              = v_loss
        self.policyLoss         = policy_loss

        self.scaler             = torch.cuda.amp.GradScaler()
        self.device             = set_device(self.use_gpu)
        self.i_update           = 0
        
        self.soft_q1_optimizer  = Adam(self.soft_q1.parameters(), lr = learning_rate)
        self.soft_q2_optimizer  = Adam(self.soft_q2.parameters(), lr = learning_rate)
        self.value_optimizer    = Adam(self.value.parameters(), lr = learning_rate)
        self.policy_optimizer   = Adam(self.policy.parameters(), lr = learning_rate)  

        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(param.data)

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.save_eps(state, action, reward, done, next_state)

    def save_memory(self, policy_memory):
        states, actions, rewards, dones, next_states = policy_memory.get_all_items()
        self.memory.save_all(states, actions, rewards, dones, next_states)

    def __training_q(self, states, actions, rewards, dones, next_states, q_net, q_optimizer):
        q_optimizer.zero_grad()

        predicted_q_value   = q_net(states, actions)
        next_value          = self.target_value(next_states)

        loss = self.qLoss.compute_loss(predicted_q_value, rewards, dones, next_value)        
        loss.backward()

        q_optimizer.step()        

    def __training_values(self, states):
        self.value_optimizer.zero_grad()

        action_datas        = self.policy(states)
        actions             = self.distribution.sample(action_datas)

        q_value1            = self.soft_q1(states, actions)
        q_value2            = self.soft_q2(states, actions)
        predicted_value     = self.value(states)

        loss = self.vLoss.compute_loss(predicted_value, action_datas, actions, q_value1, q_value2)
        loss.backward()

        self.value_optimizer.step()

    def __training_policy(self, states):
        self.policy_optimizer.zero_grad()

        action_datas    = self.policy(states)
        actions         = self.distribution.sample(action_datas)

        q_value1        = self.soft_q1(states, actions)
        q_value2        = self.soft_q2(states, actions)

        loss = self.policyLoss.compute_loss(action_datas, actions, q_value1, q_value2)
        loss.backward()

        self.policy_optimizer.step()
        
    def act(self, state):
        state               = to_tensor(state, use_gpu = self.use_gpu, first_unsqueeze = True, detach = True)
        action_datas        = self.policy(state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.act_deterministic(action_datas)
              
        return to_numpy(action, self.use_gpu)

    def update(self):
        if len(self.memory) > self.batch_size:
            for _ in range(self.epochs):
                dataloader  = DataLoader(self.memory, self.batch_size, shuffle = True)
                states, actions, rewards, dones, next_states = next(iter(dataloader))

                self.__training_q(to_tensor(states, use_gpu = self.use_gpu), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu), self.soft_q1, self.soft_q1_optimizer)

                self.__training_q(to_tensor(states, use_gpu = self.use_gpu), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu), self.soft_q2, self.soft_q2_optimizer)

                self.__training_values(to_tensor(states, use_gpu = self.use_gpu))
                self.__training_policy(to_tensor(states, use_gpu = self.use_gpu))

            for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def save_weights(self):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.ppo_optimizer.state_dict()
            }, self.folder + '/policy.tar')
        
        torch.save({
            'model_state_dict': self.value.state_dict(),
            'optimizer_state_dict': self.aux_optimizer.state_dict()
            }, self.folder + '/value.tar')
        
    def load_weights(self, device = None):
        if device == None:
            device = self.device

        policy_checkpoint = torch.load(self.folder + '/policy.tar', map_location = device)
        self.policy.load_state_dict(policy_checkpoint['model_state_dict'])
        self.ppo_optimizer.load_state_dict(policy_checkpoint['optimizer_state_dict'])

        value_checkpoint = torch.load(self.folder + '/value.tar', map_location = device)
        self.value.load_state_dict(value_checkpoint['model_state_dict'])
        self.aux_optimizer.load_state_dict(value_checkpoint['optimizer_state_dict'])
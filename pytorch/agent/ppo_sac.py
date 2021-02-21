import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.pytorch_utils import set_device, to_numpy, to_tensor

class AgentPPOSAC():
    def __init__(self, Policy_Model, Value_Model, Q_Model, state_dim, action_dim, distribution, on_policy_loss, aux_loss, q_loss, v_loss, off_policy_loss, 
                on_policy_memory, aux_memory, off_memory, is_training_mode = True, policy_kl_range = 0.03, policy_params = 5, value_clip = 1.0, 
                entropy_coef = 0.0, vf_loss_coef = 1.0, soft_tau = 0.95, batch_size = 32, PPO_epochs = 10, Aux_epochs = 10, SAC_epochs = 1, 
                gamma = 0.99, lam = 0.95, learning_rate = 3e-4, folder = 'model', use_gpu = True, n_aux_update = 10):

        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.soft_tau           = soft_tau
        self.batch_size         = batch_size  
        self.PPO_epochs         = PPO_epochs
        self.Aux_epochs         = Aux_epochs
        self.SAC_epochs         = SAC_epochs
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.state_dim          = state_dim
        self.learning_rate      = learning_rate
        self.folder             = folder
        self.use_gpu            = use_gpu
        self.n_aux_update       = n_aux_update

        self.policy             = Policy_Model(state_dim, action_dim, self.use_gpu)
        self.policy_old         = Policy_Model(state_dim, action_dim, self.use_gpu)

        self.value              = Value_Model(state_dim, self.use_gpu)
        self.value_old          = Value_Model(state_dim, self.use_gpu)

        self.soft_q1            = Q_Model(state_dim, action_dim, self.use_gpu)
        self.soft_q2            = Q_Model(state_dim, action_dim, self.use_gpu)

        self.distribution       = distribution

        self.onPolicyMemory     = on_policy_memory
        self.auxMemory          = aux_memory
        self.offMemory          = off_memory
        
        self.onPolicyLoss       = on_policy_loss
        self.auxLoss            = aux_loss
        self.qLoss              = q_loss
        self.vLoss              = v_loss
        self.offPolicyLoss      = off_policy_loss        

        self.scaler             = torch.cuda.amp.GradScaler()
        self.device             = set_device(self.use_gpu)
        self.i_update           = 0
        
        self.soft_q_optimizer   = Adam(self.soft_q.parameters(), lr = learning_rate)
        self.value_optimizer    = Adam(self.value.parameters(), lr = learning_rate)
        self.policy_optimizer   = Adam(self.policy.parameters(), lr = learning_rate)
        
        self.ppo_optimizer      = Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr = learning_rate)        
        self.aux_optimizer      = Adam(self.policy.parameters(), lr = learning_rate)  

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.save_eps(state, action, reward, done, next_state)

    def save_memory(self, policy_memory):
        states, actions, rewards, dones, next_states = policy_memory.get_all_items()
        self.memory.save_all(states, actions, rewards, dones, next_states)

    def __training_off_q(self, states, actions, rewards, dones, next_states, q_net, q_optimizer):
        q_optimizer.zero_grad()

        predicted_q_value   = q_net(states, actions)
        next_value          = self.value(next_states)

        with torch.cuda.amp.autocast():
            loss = self.qLoss.compute_loss(predicted_q_value, rewards, dones, next_value)

        self.scaler.scale(loss).backward()
        self.scaler.step(q_optimizer)
        self.scaler.update()

    def __training_off_values(self, states):
        self.value_optimizer.zero_grad()

        action_datas, _     = self.policy(states)
        actions             = self.distribution.sample(action_datas)

        q_value1            = self.soft_q1(states, actions)
        q_value2            = self.soft_q2(states, actions)
        predicted_value     = self.value(states)

        with torch.cuda.amp.autocast():
            loss = self.vLoss.compute_loss(predicted_value, action_datas, actions, q_value1, q_value2)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.value_optimizer)
        self.scaler.update()

    def __training_off_policy(self, states):
        self.policy_optimizer.zero_grad()

        action_datas, _ = self.policy(states)
        actions         = self.distribution.sample(action_datas)

        q_value1        = self.soft_q1(states, actions)
        q_value2        = self.soft_q2(states, actions)

        with torch.cuda.amp.autocast():
            loss = self.policyLoss.compute_loss(action_datas, actions, q_value1, q_value2)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.policy_optimizer)
        self.scaler.update()

    def __training_on_policy(self, states, actions, rewards, dones, next_states):         
        self.ppo_optimizer.zero_grad()

        action_datas, _     = self.policy(states)
        values              = self.value(states)
        old_action_datas, _ = self.policy_old(states, True)
        old_values          = self.value_old(states, True)
        next_values         = self.value(next_states, True)

        with torch.cuda.amp.autocast():
            ppo_loss    = self.policyLoss.compute_loss(action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones)

        self.scaler.scale(ppo_loss).backward()
        self.scaler.step(self.ppo_optimizer)
        self.scaler.update()

    def __training_on_aux(self, states):        
        self.aux_optimizer.zero_grad()
        
        action_datas, values    = self.policy(states)
        returns                 = self.value(states, True)
        old_action_datas, _     = self.policy_old(states, True)

        with torch.cuda.amp.autocast():
            joint_loss  = self.auxLoss.compute_loss(action_datas, old_action_datas, values, returns)

        self.scaler.scale(joint_loss).backward()
        self.scaler.step(self.aux_optimizer)
        self.scaler.update()

    def __update_ppo(self):
        dataloader = DataLoader(self.policy_memory, self.batch_size, shuffle = False)

        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in dataloader:
                self.__training_on_policy(to_tensor(states, use_gpu = self.use_gpu), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu))

        states, _, _, _, _ = self.policy_memory.get_all_items()
        self.auxMemory.save_all(states)
        self.onPolicyMemory.clear_memory()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())    

    def __update_aux(self):
        dataloader  = DataLoader(self.aux_memory, self.batch_size, shuffle = False)

        for _ in range(self.Aux_epochs):       
            for states in dataloader:
                self.__training_on_aux(to_tensor(states, use_gpu = self.use_gpu))

        self.auxMemory.clear_memory()
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def act(self, state):
        state               = to_tensor(state, use_gpu = self.use_gpu, first_unsqueeze = True, detach = True)
        action_datas        = self.policy(state)
        
        if self.is_training_mode:
            action = self.distribution.sample(action_datas)
        else:
            action = self.distribution.act_deterministic(action_datas)
              
        return to_numpy(action, self.use_gpu)

    def update_off(self):
        dataloader  = DataLoader(self.offMemory, self.batch_size, shuffle = False)

        for _ in range(self.SAC_epochs):       
            for states, actions, rewards, dones, next_states in dataloader:
                self.__training_off_q(to_tensor(states, use_gpu = self.use_gpu), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu), self.soft_q1, self.soft_q1_optimizer)

                self.__training_off_q(to_tensor(states, use_gpu = self.use_gpu), actions.float().to(self.device), rewards.float().to(self.device), 
                    dones.float().to(self.device), to_tensor(next_states, use_gpu = self.use_gpu), self.soft_q2, self.soft_q2_optimizer)

                self.__training_off_values(to_tensor(states, use_gpu = self.use_gpu))
                self.__training_off_policy(to_tensor(states, use_gpu = self.use_gpu))

        self.offMemory.clear_memory()

    def update_on(self):
        self.__update_ppo()
        self.i_update += 1

        if self.i_update % self.n_aux_update == 0:
            self.__update_aux()
            self.i_update = 0

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

        if self.is_training_mode:
            self.policy.train()
            self.value.train()
            print('Model is training...')

        else:
            self.policy.eval()
            self.value.eval()
            print('Model is evaluating...')
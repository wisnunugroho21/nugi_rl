import torch

class SAC():
    def __init__(self, Policy_Model, Value_Model, state_dim, action_dim, distribution, policy_loss, aux_loss, policy_memory, aux_memory,
                is_training_mode = True, policy_kl_range = 0.03, policy_params = 5, value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, 
                batch_size = 32, PPO_epochs = 10, Aux_epochs = 10, gamma = 0.99, lam = 0.95, 
                learning_rate = 3e-4, folder = 'model', use_gpu = True, n_aux_update = 10):   

        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.batch_size         = batch_size  
        self.PPO_epochs         = PPO_epochs
        self.Aux_epochs         = Aux_epochs
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.state_dim          = state_dim
        self.learning_rate      = learning_rate
        self.folder             = folder
        self.use_gpu            = use_gpu
        self.n_aux_update       = n_aux_update

        self.policy             = Policy_Model(state_dim, action_dim, self.use_gpu).float().to(set_device(use_gpu))
        self.policy_old         = Policy_Model(state_dim, action_dim, self.use_gpu).float().to(set_device(use_gpu))

        self.value              = Value_Model(state_dim, action_dim).float().to(set_device(use_gpu))
        self.value_old          = Value_Model(state_dim, action_dim).float().to(set_device(use_gpu))

        self.distribution       = distribution
        self.policy_memory      = policy_memory
        self.aux_memory         = aux_memory
        
        self.trulyPPO           = policy_loss
        self.auxLoss            = aux_loss      

        self.scaler             = torch.cuda.amp.GradScaler()
        self.device             = set_device(self.use_gpu)
        self.i_update           = 0

        self.ppo_optimizer      = Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr = learning_rate)        
        self.aux_optimizer      = Adam(self.policy.parameters(), lr = learning_rate)  

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()

    def save_eps(self, state, action, reward, done, next_state):
        self.policy_memory.save_eps(state, action, reward, done, next_state)

    def save_memory(self, policy_memory):
        states, actions, rewards, dones, next_states = policy_memory.get_all_items()
        self.policy_memory.save_all(states, actions, rewards, dones, next_states)
        
    def act(self, state):
        pass

    def update(self):
        pass

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
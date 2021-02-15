import torch

from distribution.basic import BasicDiscrete, BasicContinous

from agent.agent import Agent
from loss.truly_ppo import TrulyPPO
from loss.joint_aux import JointAux
from utils.pytorch_utils import set_device, to_numpy

from agent.agent_standard import AgentContinous
from memory.image_state_memory import ImageStateMemory
from memory.image_state_aux_memory import ImageStateAuxMemory

from torch.utils.data import Dataset, DataLoader

class AgentImageStates(AgentContinous):
    def __init__(self, Policy_Model, Value_Model, state_dim, action_dim,
                is_training_mode = True, policy_kl_range = 0.03, policy_params = 5, 
                value_clip = 1.0, entropy_coef = 0.0, vf_loss_coef = 1.0, 
                batch_size = 32, PPO_epochs = 10, Aux_epochs = 10, gamma = 0.99,
                lam = 0.95, learning_rate = 3e-4, action_std = 1.0, folder = 'model', use_gpu = True):
        
        super(AgentImageStates, self).__init__(Policy_Model, Value_Model, state_dim, action_dim,
                is_training_mode, policy_kl_range, policy_params, 
                value_clip, entropy_coef, vf_loss_coef, 
                batch_size, PPO_epochs, Aux_epochs, gamma,
                lam , learning_rate, action_std, folder, use_gpu)

        self.policy_memory  = ImageStateMemory()
        self.aux_memory     = ImageStateAuxMemory()

    def save_eps(self, image, state, action, reward, done, next_image, next_state):
        self.policy_memory.save_eps(image, state, action, reward, done, next_image, next_state)

    def save_all(self, images, states, actions, rewards, dones, next_images, next_states):
        self.policy_memory.save_all(images, states, actions, rewards, dones, next_images, next_states)

    def save_memory(self, memory):
        images, states, actions, rewards, dones, next_images, next_states = memory.get_all_items()
        self.policy_memory.save_all(images, states, actions, rewards, dones, next_images, next_states)

    def act(self, image, state):
        image           = torch.FloatTensor(image).to(self.device)
        state           = torch.FloatTensor(state).to(self.device)

        image           = image.unsqueeze(0).detach() if len(image.shape) == 3 else image.detach()
        state           = state.unsqueeze(0).detach() if len(state.shape) == 1 else state.detach()

        action_mean, _  = self.policy(image, state)
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action = self.distribution.sample(action_mean, self.action_std)
        else:
            action = action_mean.detach()  
              
        return to_numpy(action.squeeze(0), self.use_gpu)

    # Get loss and Do backpropagation
    def training_ppo(self, images, states, actions, rewards, dones, next_images, next_states):
        self.ppo_optimizer.zero_grad()

        action_mean, _      = self.policy(images, states)
        values              = self.value(images, states)
        old_action_mean, _  = self.policy_old(images, states)
        old_values          = self.value_old(images, states)
        next_values         = self.value(next_images, next_states)

        with torch.cuda.amp.autocast():
            ppo_loss    = self.trulyPPO.compute_continous_loss(action_mean, self.action_std, old_action_mean, self.action_std, values, old_values, next_values, actions, rewards, dones)

        self.scaler.scale(ppo_loss).backward()
        self.scaler.step(self.ppo_optimizer)
        self.scaler.update()

    def training_aux(self, images, states):
        self.aux_optimizer.zero_grad()

        returns             = self.value(images, states)
        action_mean, values = self.policy(images, states)
        old_action_mean, _  = self.policy_old(images, states)

        with torch.cuda.amp.autocast():
            joint_loss  = self.auxLoss.compute_continous_loss(action_mean, self.action_std, old_action_mean, self.action_std, values, returns)

        self.scaler.scale(joint_loss).backward()
        self.scaler.step(self.aux_optimizer)
        self.scaler.update()

    def update_ppo(self, policy_memory = None, aux_memory = None):
        if policy_memory is None:
            policy_memory = self.policy_memory

        if aux_memory is None:
            aux_memory = self.aux_memory

        dataloader = DataLoader(policy_memory, self.batch_size, shuffle = False)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for images, states, actions, rewards, dones, next_images, next_states in dataloader: 
                self.training_ppo(images.float().to(self.device), states.float().to(self.device), actions.float().to(self.device), \
                    rewards.float().to(self.device), dones.float().to(self.device), next_images.float().to(self.device), next_states.float().to(self.device))

        # Clear the memory
        images, states, _, _, _, _, _ = policy_memory.get_all_items()
        aux_memory.save_all(images, states)
        policy_memory.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

        return policy_memory, aux_memory

    def update_aux(self, aux_memory = None):
        if aux_memory is None:
            aux_memory = self.aux_memory

        dataloader  = DataLoader(aux_memory, self.batch_size, shuffle = False)

        # Optimize policy for K epochs:
        for _ in range(self.Aux_epochs):       
            for images, states in dataloader:
                self.training_aux(images.float().to(self.device), states.float().to(self.device))

        # Clear the memory
        aux_memory.clear_memory()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        return aux_memory
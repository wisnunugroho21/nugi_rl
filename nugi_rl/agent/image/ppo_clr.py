import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import device
from torchvision.transforms import Compose

from copy import deepcopy

from nugi_rl.distribution.base import Distribution
from nugi_rl.agent.standard.ppo import AgentPPO
from nugi_rl.loss.ppo.base import Ppo
from nugi_rl.loss.value import ValueLoss
from nugi_rl.loss.entropy import EntropyLoss
from nugi_rl.loss.clr.base import CLR
from nugi_rl.memory.policy.base import Memory
from nugi_rl.memory.clr import ClrMemory

class AgentImagePpoClr(AgentPPO):
    def __init__(self, policy: Module, value: Module, cnn: Module, projector: Module, distribution: Distribution, policy_loss: Ppo, value_loss: ValueLoss, entropy_loss: EntropyLoss, aux_clr_loss: CLR, 
        memory: Memory, aux_clr_memory: ClrMemory, optimizer: Optimizer, aux_clr_optimizer: Optimizer, trans: Compose, ppo_epochs: int = 10, aux_clr_epochs = 5, is_training_mode: bool = True, 
        batch_size: int = 32, folder: str = 'model', device: device = torch.device('cuda'), policy_old: Module = None, value_old: Module = None, cnn_old: Module = None, projector_old: Module = None):
        
        super().__init__(policy, value, distribution, policy_loss, value_loss, entropy_loss, memory, optimizer, ppo_epochs = ppo_epochs, is_training_mode = is_training_mode, batch_size = batch_size, folder = folder, device = device, policy_old = policy_old, value_old = value_old)

        self.cnn                = cnn
        self.projector          = projector

        self.cnn_old            = cnn_old
        self.projector_old      = projector_old

        self.trans              = trans

        self.aux_clr_loss       = aux_clr_loss
        self.aux_clr_memory     = aux_clr_memory
        self.aux_clr_optimizer  = aux_clr_optimizer
        self.aux_clr_epochs     = aux_clr_epochs

        if self.cnn_old is None:
            self.cnn_old = deepcopy(self.cnn)

        if self.projector_old is None:
            self.projector_old  = deepcopy(self.projector)
    
    def _update_step(self, states: list, actions: list, rewards: float, dones: bool, next_states: list) -> None:
        self.optimizer.zero_grad()

        res                 = self.cnn(states)

        action_datas        = self.policy(res)
        values              = self.value(res)

        res_old             = self.cnn_old(states, True)

        old_action_datas    = self.policy_old(res_old, True)
        old_values          = self.value_old(res_old, True)

        next_res            = self.cnn(next_states, True)
        next_values         = self.value(next_res, True)

        loss = self.policy_loss.compute_loss(action_datas, old_action_datas, values, next_values, actions, rewards, dones) + \
            self.value_loss.compute_loss(values, next_values, rewards, dones, old_values) + \
            self.entropy_loss.compute_loss(action_datas)
        
        loss.backward()
        self.optimizer.step()

    def _update_step_aux_clr(self, input_images, target_images):
        self.aux_clr_optimizer.zero_grad()

        res_anchor        = self.cnn(input_images)
        encoded_anchor    = self.projector(res_anchor)

        res_target        = self.cnn_old(target_images, True)
        encoded_target    = self.projector_old(res_target, True)

        loss = self.aux_clr_loss.compute_loss(encoded_anchor, encoded_target)

        loss.backward()
        self.aux_clr_optimizer.step()

    def _update_ppo(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

        for _ in range(self.ppo_epochs):
            dataloader = DataLoader(self.memory, self.batch_size, shuffle = False)
            for states, actions, rewards, dones, next_states in dataloader:
                self._update_step(states.to(self.device), actions.to(self.device), rewards.to(self.device), dones.to(self.device), next_states.to(self.device))

        states, _, _, _, _ = self.memory.get()
        self.aux_clr_memory.save_all(states)
        self.memory.clear()

    def _update_aux_clr(self):
        self.cnn_old.load_state_dict(self.cnn.state_dict())
        self.projector_old.load_state_dict(self.projector.state_dict())

        for _ in range(self.aux_clr_epochs):
            dataloader  = DataLoader(self.aux_clr_memory, self.batch_size, shuffle = True, num_workers = 8)
            for input_images, target_images in dataloader:
                self._update_step_aux_clr(input_images.to(self.device), target_images.to(self.device))            

        self.aux_clr_memory.clear()

    def act(self, state) -> list:
        with torch.inference_mode():
            state           = self.trans(state).to(self.device).unsqueeze(0)          
            action_datas    = self.policy(state)
            
            if self.is_training_mode:
                action = self.distribution.sample(action_datas)
            else:
                action = self.distribution.deterministic(action_datas)

            action = action.squeeze(0).detach().tolist()
              
        return action

    def logprob(self, state, action: list) -> list:
        with torch.inference_mode():
            state           = self.trans(state).to(self.device).unsqueeze(0)
            action          = torch.FloatTensor(action).float().to(self.device).unsqueeze(0)

            action_datas    = self.policy(state)

            logprobs        = self.distribution.logprob(action_datas, action)
            logprobs        = logprobs.squeeze(0).detach().tolist()

        return logprobs

    def save_obs(self, state: list, action: list, reward: float, done: bool, next_state: list) -> None:
        self.memory.save(state, action, reward, done, next_state)
        
    def update(self) -> None:
        self._update_ppo()
        self._update_aux_clr()

    def get_obs(self, start_position: int = None, end_position: int = None) -> tuple:
        self.memory.get(start_position, end_position)

    def load_weights(self) -> None:
        model_checkpoint = torch.load(self.folder + '/ppg.pth', map_location = self.device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(model_checkpoint['ppo_optimizer_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.value.train()

        else:
            self.policy.eval()
            self.value.eval()

    def save_weights(self) -> None:            
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'ppo_optimizer_state_dict': self.optimizer.state_dict(),
        }, self.folder + '/ppg.pth')    
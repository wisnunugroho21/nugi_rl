import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch import device
from torchvision.transforms import Compose

from nugi_rl.agent.standard.ppo import AgentPPO
from nugi_rl.distribution.base import Distribution
from nugi_rl.loss.ppo.base import Ppo
from nugi_rl.loss.value import ValueLoss
from nugi_rl.loss.entropy import EntropyLoss
from nugi_rl.memory.base import Memory

class AgentImagePPO(AgentPPO):  
    def __init__(self, policy: Module, value: Module, distribution: Distribution, policy_loss: Ppo, value_loss: ValueLoss, entropy_loss: EntropyLoss, memory: Memory, optimizer: Optimizer, trans: Compose, ppo_epochs: int = 10, is_training_mode: bool = True, batch_size: int = 32, folder: str = 'model', device: device = torch.device('cuda'), policy_old: Module = None, value_old: Module = None):
        super().__init__(policy, value, distribution, policy_loss, value_loss, entropy_loss, memory, optimizer, ppo_epochs=ppo_epochs, is_training_mode=is_training_mode, batch_size=batch_size, folder=folder, device=device, policy_old=policy_old, value_old=value_old)
        self.trans = trans

    def act(self, state) -> Tensor:
        with torch.inference_mode():
            state           = self.trans(state).to(self.device).unsqueeze(0)
            action_datas    = self.policy(state)
            
            if self.is_training_mode:
                action = self.distribution.sample(action_datas)
            else:
                action = self.distribution.deterministic(action_datas)

            action = action.squeeze(0).detach()
              
        return action

    def logprobs(self, state, action: Tensor) -> Tensor:
        with torch.inference_mode():
            state           = self.trans(state).to(self.device).unsqueeze(0)
            action          = action.unsqueeze(0)

            action_datas    = self.policy(state)

            logprobs        = self.distribution.logprob(action_datas, action)
            logprobs        = logprobs.squeeze(0).detach()

        return logprobs
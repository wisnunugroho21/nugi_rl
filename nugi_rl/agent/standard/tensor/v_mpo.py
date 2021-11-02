import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch import device

from nugi_rl.distribution.base import Distribution
from nugi_rl.loss.v_mpo.alpha.base import AlphaLoss
from nugi_rl.loss.v_mpo.phi_loss import PhiLoss
from nugi_rl.loss.v_mpo.temperature_loss import TemperatureLoss
from nugi_rl.loss.value import ValueLoss
from nugi_rl.loss.entropy import EntropyLoss
from nugi_rl.memory.base import Memory

from nugi_rl.agent.standard.v_mpo import AgentVMPO

class AgentTensorVMPO(AgentVMPO):
    def __init__(self, policy: Module, value: Module, distribution: Distribution, alpha_loss: AlphaLoss, phi_loss: PhiLoss, entropy_loss: EntropyLoss, temperature_loss: TemperatureLoss, value_loss: ValueLoss, memory: Memory, policy_optimizer: Optimizer, value_optimizer: Optimizer, epochs: int = 10, is_training_mode: bool = True, batch_size: int = 64, folder: str = 'model', device: device = ..., old_policy: Module = None, old_value: Module = None):
        super().__init__(policy, value, distribution, alpha_loss, phi_loss, entropy_loss, temperature_loss, value_loss, memory, policy_optimizer, value_optimizer, epochs=epochs, is_training_mode=is_training_mode, batch_size=batch_size, folder=folder, device=device, old_policy=old_policy, old_value=old_value)

    def act(self, state: Tensor) -> Tensor:
        with torch.inference_mode():
            state               = state.unsqueeze(0)
            action_datas, _, _  = self.policy(state)
            
            if self.is_training_mode:
                action = self.distribution.sample(action_datas)
            else:
                action = self.distribution.deterministic(action_datas)

            action  = action.squeeze(0).detach()
              
        return action

    def logprobs(self, state: Tensor, action: Tensor) -> Tensor:
        with torch.inference_mode():
            state               = state.unsqueeze(0)
            action              = action.unsqueeze(0)

            action_datas, _, _  = self.policy(state)

            logprobs            = self.distribution.logprob(action_datas, action)
            logprobs            = logprobs.squeeze(0).detach()

        return logprobs
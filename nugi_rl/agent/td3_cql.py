import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch import device

from nugi_rl.agent.td3 import AgentTd3
from nugi_rl.loss.td3.policy_loss import PolicyLoss
from nugi_rl.loss.td3.q_loss import QLoss
from nugi_rl.loss.cql_regularizer import CqlRegularizer
from nugi_rl.memory.policy.base import PolicyMemory

class AgentTd3Cql(AgentTd3):
    def __init__(self, soft_q1: Module, soft_q2: Module, policy: Module, q_loss: QLoss, policy_loss: PolicyLoss, cql_reg_loss: CqlRegularizer, 
        memory: PolicyMemory, soft_q_optimizer: Optimizer, policy_optimizer: Optimizer, is_training_mode: bool = True, batch_size: int = 32, epochs: int = 1, 
        soft_tau: float = 0.95, folder: str = 'model', device: device = torch.device('cuda:0'), target_q1: Module = None, target_q2: Module = None, dont_unsqueeze = False) -> None:

        super().__init__(soft_q1, soft_q2, policy, q_loss, policy_loss, memory, soft_q_optimizer, policy_optimizer, is_training_mode,
            batch_size, epochs, soft_tau, folder, device, target_q1, target_q2, dont_unsqueeze)

        self.cqlRegLoss = cql_reg_loss

    def _update_step_q(self, states: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor, next_states: Tensor):
        self.soft_q_optimizer.zero_grad()        

        predicted_q1        = self.soft_q1(states, actions)
        predicted_q2        = self.soft_q2(states, actions)

        predicted_actions   = self.policy(states, True)
        naive_q1_value      = self.soft_q1(states, predicted_actions)
        naive_q2_value      = self.soft_q2(states, predicted_actions)

        next_actions        = self.policy(next_states, True)
        target_next_q1      = self.target_q1(next_states, next_actions, True)
        target_next_q2      = self.target_q2(next_states, next_actions, True)

        loss  = self.qLoss(predicted_q1, predicted_q2, target_next_q1, target_next_q2, rewards, dones) + \
            self.cqlRegLoss(predicted_q1, predicted_q2, naive_q1_value, naive_q2_value)

        loss.backward()
        self.soft_q_optimizer.step()
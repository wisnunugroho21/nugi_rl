import torch
from agent.standard.v_mpo import AgentVMPO

class AgentTensorVMPO(AgentVMPO):
    def __init__(self, policy, value, distribution, alpha_loss, phi_loss, entropy_loss, temperature_loss, value_loss, 
            policy_memory, policy_optimizer, value_optimizer, policy_epochs=1, is_training_mode=True, batch_size=32, folder='model', device=..., old_policy=None, old_value=None):
        
        super().__init__(policy, value, distribution, alpha_loss, phi_loss, entropy_loss, temperature_loss, value_loss, 
            policy_memory, policy_optimizer, value_optimizer, policy_epochs=policy_epochs, is_training_mode=is_training_mode, batch_size=batch_size, folder=folder, device=device, old_policy=old_policy, old_value=old_value)

    def act(self, state):
        with torch.inference_mode():
            state               = state.unsqueeze(0)
            action_datas, _, _  = self.policy(state)
            
            if self.is_training_mode:
                action = self.distribution.sample(action_datas)
            else:
                action = self.distribution.deterministic(action_datas)

            action  = action.squeeze(0).detach()
              
        return action

    def logprobs(self, state, action):
        with torch.inference_mode():
            state               = state.unsqueeze(0)
            action              = action.unsqueeze(0)

            action_datas, _, _  = self.policy(state)

            logprobs            = self.distribution.logprob(action_datas, action)
            logprobs            = logprobs.squeeze(0).detach()

        return logprobs

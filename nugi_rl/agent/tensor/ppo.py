
import torch
from agent.standard.ppo import AgentPPO

class AgentRTensorPPO(AgentPPO):  
    def __init__(self, policy, value, distribution, ppo_loss, ppo_memory, optimizer, ppo_epochs = 10, is_training_mode = True, 
        batch_size = 32,  folder = 'model', device = torch.device('cuda:0'), policy_old = None, value_old = None): 

        super().__init__(policy, value, distribution, ppo_loss, ppo_memory, optimizer, ppo_epochs, is_training_mode, 
            batch_size,  folder, device, policy_old, value_old)

    def act(self, state):
        with torch.inference_mode():
            state           = state.unsqueeze(0)
            action_datas    = self.policy(state)
            
            if self.is_training_mode:
                action = self.distribution.sample(action_datas)
            else:
                action = self.distribution.deterministic(action_datas)

            action = action.squeeze(0).detach()
              
        return action

    def logprobs(self, state, action):
        with torch.inference_mode():
            state           = state.unsqueeze(0)
            action          = action.unsqueeze(0)

            action_datas    = self.policy(state)

            logprobs        = self.distribution.logprob(action_datas, action)
            logprobs        = logprobs.squeeze(0).detach()

        return logprobs

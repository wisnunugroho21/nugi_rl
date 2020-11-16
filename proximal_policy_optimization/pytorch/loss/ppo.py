class PPO():
    
    # Loss for PPO  
    def compute_discrete_loss(self, action_probs, old_action_probs, values, old_values, next_values, actions, rewards, dones):
        pass

    def compute_continous_loss(self, action_mean, action_std, old_action_mean, old_action_std, values, old_values, next_values, actions, rewards, dones):    
        pass
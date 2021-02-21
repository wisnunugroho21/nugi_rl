from policy_function.advantage_function import AdvantageFunction

class QLoss():
    def __init__(self, gamma = 0.99, lam = 0.95):
        self.advantage_function = AdvantageFunction(gamma, lam)

    def compute_loss(self, predicted_q_value, rewards, dones, next_values):
        advantages      = self.advantage_function.generalized_advantage_estimation(rewards, predicted_q_value, next_values, dones)
        target_q_value  = (advantages + predicted_q_value).detach()

        q_value_loss    = ((target_q_value - predicted_q_value).pow(2) * 0.5).mean()
        return q_value_loss
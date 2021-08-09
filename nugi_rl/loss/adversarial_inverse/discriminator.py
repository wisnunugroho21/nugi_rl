class DiscriminatorLoss():
    def __init__(self, distribution, gamma) -> None:
        self.distribution   = distribution
        self.gamma          = gamma

    def compute_descrimination(self, g_values, h_values, h_next_values, logprobs, dones):
            f_values        = g_values + (1.0 - dones) * self.gamma * h_next_values - h_values
            descrimination  = f_values.exp() / (f_values.exp() + logprobs.exp())

            return descrimination

    def compute_loss(self, expert_g_values, expert_h_values, expert_h_next_values, expert_logprobs, expert_dones,
        policy_g_values, policy_h_values, policyt_h_next_values, policy_logprobs, policy_dones):

        expert_descrimination = self.compute_descrimination(expert_g_values, expert_h_values, expert_h_next_values, expert_logprobs, expert_dones)
        policy_descrimination = self.compute_descrimination(policy_g_values, policy_h_values, policyt_h_next_values, policy_logprobs, policy_dones)

        loss = -1 * (expert_descrimination.log() + (1 - policy_descrimination).log())
        return loss.mean()
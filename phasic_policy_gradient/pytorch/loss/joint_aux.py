import torch

from distribution.basic import BasicDiscrete, BasicContinous
from policy_function.advantage_function import AdvantageFunction

class JointAux():
    def __init__(self, device, value_clip):
        self.discrete           = BasicDiscrete(device)
        self.continous          = BasicContinous(device)
        self.advantagefunction  = AdvantageFunction()

        self.value_clip         = value_clip

    def compute_discrete_loss(self, action_probs, old_action_probs, values, Returns):        
        # Don't use old value in backpropagation
        Old_action_probs    = old_action_probs.detach()
        Returns             = Returns.detach()

        # Finding KL Divergence                
        Kl                  = self.discrete.kldivergence(Old_action_probs, action_probs).mean()
        aux_loss            = ((Returns - values).pow(2) * 0.5).mean()

        return aux_loss + Kl

    def compute_continous_loss(self, action_mean, action_std, old_action_mean, old_action_std, values, Returns):
        # Don't use old value in backpropagation
        Old_action_mean = old_action_mean.detach()

        # Finding KL Divergence                
        Kl              = self.continous.kldivergence(Old_action_mean, old_action_std, action_mean, action_std).mean()
        aux_loss        = ((Returns - values).pow(2) * 0.5).mean()

        return aux_loss + Kl
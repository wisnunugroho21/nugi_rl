from torch.nn import Module

def copy_parameters(source_model: Module, target_model: Module, tau: int = 0.95):
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))

    return target_model

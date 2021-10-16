from torch.tensor import Tensor

class PPO(): 
    def compute_loss(self, action_datas: tuple, old_action_datas: tuple, values: Tensor, old_values: Tensor, next_values: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor) -> Tensor:
        raise NotImplementedError
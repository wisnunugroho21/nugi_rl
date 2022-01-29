import wandb
from torch.nn import Module

from nugi_rl.utilities.plotter.base import Plotter

class WeightBiasPlotter(Plotter):
    def __init__(self, config: dict, project_name: str, entity: str) -> None:
        wandb.config = config
        wandb.init(project = project_name, entity = entity)        

    def plot(self, datas: dict, model: Module = None) -> None:
        wandb.log(datas)

        if model is not None:
            wandb.watch(model)
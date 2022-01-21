import wandb
from torch.nn import Module

from nugi_rl.helpers.plotter.base import Plotter

class WeightBiasPlotter(Plotter):
    def __init__(self, config: dict, project_name: str) -> None:
        wandb.init(config = config, project = project_name)

    def plot(self, datas: dict, model: Module = None) -> None:
        wandb.log(datas)

        if model is not None:
            wandb.watch(model)
from torch.utils.tensorboard import SummaryWriter

from nugi_rl.utilities.plotter.base import Plotter

class TensorboardPlotter(Plotter):
    def __init__(self, summary_writer: SummaryWriter) -> None:
        self.summary_writer = summary_writer

    def plot(self, datas: dict) -> None:
        for key, value in datas.items():
            self.summary_writer.add_scalar(key, value)
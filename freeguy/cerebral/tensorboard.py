from torch.utils.tensorboard import SummaryWriter
import torch 

class GetSummaryWriter:
    """
    "summary" : {
        "name": "experiment",
    }
    """
    def __init__(self, args):
        self.writer = SummaryWriter(log_dir=[args.summary["name"]])

    def insert(self, name, key, value):
        self.writer.add_scalar(name, key, value)
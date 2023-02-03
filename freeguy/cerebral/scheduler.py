import torch 
from torch.optim.lr_scheduler import ReduceLROnPlateau


# scheduler = ReduceLROnPlateau(optimizer, patience=3, verbose=True)

class GetScheduler:
    """
    "scheduler" : {
        "name": "CosineAnnealingWarmRestarts",
    }
    """
    def __init__(self):
        self.scheduler_name_list = {
            "CosineAnnealingWarmRestarts": "",
            "ReduceLROnPlateau": ""
        }

    def __call__(self, args, optimizer:torch.nn.Module):
        return self.scheduler_name_list[args.scheduler["name"]](
            args, optimizer
        )

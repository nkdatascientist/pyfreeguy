import torch 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def CosineAnnealingWarmupRestarts_schedular(args, optimizer):
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=args.scheduler["first_cycle_steps"],
        cycle_mult=args.scheduler["cycle_mult"],
        max_lr=args.scheduler["max_lr"],
        min_lr=args.scheduler["min_lr"],
        warmup_steps=args.scheduler["warmup_steps"],
        gamma=args.scheduler["gamma"]
    )
    return scheduler

# scheduler = ReduceLROnPlateau(optimizer, patience=3, verbose=True)

class GetScheduler:
    """
    "scheduler" : {
        "name": "CosineAnnealingWarmRestarts",
        "first_cycle_steps": 200,
        "cycle_mult": 1.0,
        "max_lr": 0.1,
        "min_lr": 0.001,
        "warmup_steps": 50,
        "gamma": 0.5
    },
    """
    def __init__(self):
        self.scheduler_name_list = {
            "CosineAnnealingWarmRestarts": CosineAnnealingWarmupRestarts_schedular,
            "ReduceLROnPlateau": ReduceLROnPlateau
        }

    def __call__(self, args, optimizer:torch.nn.Module):
        return self.scheduler_name_list[args.scheduler["name"]](
            args, optimizer
        )

import torch 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
"""
1. StepLR: Decreases the learning rate by a factor after a specified number of epochs
2. MultiStepLR: Decreases the learning rate at specified epochs
3. ExponentialLR: Decreases the learning rate by a factor every epoch
4. CosineAnnealingLR: Decreases the learning rate using a cosine function
5. ReduceLROnPlateau: Decreases the learning rate when a specified metric has stopped improving
"""

def CosineAnnealingWarmupRestarts_schedular(args, optimizer):
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


def MultiStepLR_schedular(args, optimizer):
    """
    "scheduler" : {
        "name": "MultiStepLR",
    },
    """
    milestones = list(range(1, args.epochs, 10))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    return scheduler

def ReduceLROnPlateau_schedular(args, optimizer):
    """
    "scheduler" : {
        "name": "ReduceLROnPlateau",
    },
    """
    scheduler = ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    return scheduler

class GetScheduler:
    def __init__(self):
        self.scheduler_name_list = {
            "CosineAnnealingWarmRestarts": CosineAnnealingWarmupRestarts_schedular,
            "ReduceLROnPlateau": ReduceLROnPlateau_schedular,
            "MultiStepLR": MultiStepLR_schedular
        }

    def __call__(self, args, optimizer:torch.nn.Module):
        return self.scheduler_name_list[args.scheduler["name"]](
            args, optimizer
        )

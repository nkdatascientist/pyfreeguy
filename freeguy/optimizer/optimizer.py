
import torch 

def sgd_optimizer(args, model):
    """
    "optimizer" : {
        "name": "sgd",
        "lr": 0
        "momentum": 0
        "weight_decay": 0
    },
    """
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=args.optimizer["lr"],
        momentum=args.optimizer["momentum"],
        weight_decay=args.optimizer["weight_decay"]
    )
    return optimizer

def adam_optimizer(args, model):
    """
    "optimizer" : {
        "name": "adam",
        "lr": 0,
        "weight_decay": 0
    },
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.optimizer["lr"], weight_decay=args.optimizer["weight_decay"]
    )
    return optimizer


class GetOptimizer:
    def __init__(self):
        self.optimizer_name_list = {
            "sgd": sgd_optimizer,
            "adam": adam_optimizer
        }

    def __call__(self, args, model:torch.nn.Module):
        if args.optimizer["name"] in self.optimizer_name_list:
            return self.optimizer_name_list[args.optimizer["name"]](
                args, model
            )
        return None
        
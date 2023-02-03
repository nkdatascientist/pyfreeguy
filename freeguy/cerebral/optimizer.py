
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
        lr=args.sgd["lr"],
        momentum=args.sgd["momentum"],
        weight_decay=args.sgd["weight_decay"]
    )
    return optimizer

class GetOptimizer:
    def __init__(self):
        self.optimizer_name_list = {
            "sgd": sgd_optimizer
        }

    def __call__(self, args, model:torch.nn.Module):
        return self.optimizer_name_list[args.optimizer["name"]](
            args, model
        )

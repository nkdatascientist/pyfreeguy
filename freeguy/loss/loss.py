
import torch 

def crossentropyloss(args):
    return torch.nn.CrossEntropyLoss()

class GetLoss:
    def __init__(self):
        self.Loss_name_list = {
            "CrossEntropyLoss": crossentropyloss,
        }

    def __call__(self, args):
        print()
        return self.Loss_name_list[args.loss["name"]](
            args
        )




from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms 
from freeguy.transforms import GetTransforms
import torch, os

# https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
def imagenet_dataloader(args, train_transform=None, val_transform=None):
    """
    "root": "", 
    "dataloader":{
        "train": {
            "batch_size": 8,
            "shuffle": true,
            "num_workers": 4,
            "pin_memory": true,
            "drop_last": false,
            "transforms": true
            "persistent_workers": true
        },
        "val": {
            "batch_size": 8,
            "shuffle": false,
            "num_workers": 4,
            "pin_memory": true,
            "drop_last": false,
            "transforms": null
            "persistent_workers": true
        }
    }
    """

    train_transform = GetTransforms()(args, "train")
    val_transform = GetTransforms()(args, "val")

    train_dataset = ImageFolder(
        root=os.path.join(args.root, "train"),
        transform=train_transform
    )
    val_dataset = ImageFolder(
        root=os.path.join(args.root, "val"),
        transform=val_transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.dataloader["train"]["batch_size"],
        shuffle=args.dataloader["train"]["shuffle"],
        num_workers=args.dataloader["train"]["num_workers"],
        pin_memory=args.dataloader["train"]["pin_memory"],
        drop_last=args.dataloader["train"]["drop_last"],
        persistent_workers=args.dataloader["train"]["persistent_workers"]
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.dataloader["val"]["batch_size"],
        shuffle=args.dataloader["val"]["shuffle"],
        num_workers=args.dataloader["val"]["num_workers"],
        pin_memory=args.dataloader["val"]["pin_memory"],
        drop_last=args.dataloader["val"]["drop_last"],
        persistent_workers=args.dataloader["val"]["persistent_workers"]
    )

    return train_dataloader, val_dataloader



class GetDataloader:
    def __init__(self):
        self.dataloader_name_list = {
            "imagenet": imagenet_dataloader
        }

    def __call__(self, args):
        return self.dataloader_name_list[args.dataloader_name](
            args
        )

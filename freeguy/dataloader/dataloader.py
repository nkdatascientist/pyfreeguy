from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms 
from freeguy.transforms import GetTransforms
import torch

# https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
def imagenet_dataloader(args, train_transform, val_transform):
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
    train_transform = None
    val_transform = None
    if args.dataloader["train"]["transforms"]: 
        train_transform = GetTransforms(args, "train")
    if args.dataloader["val"]["transforms"]: 
        val_transform = GetTransforms(args, "val")

    train_dataset = ImageFolder(
        root=args.root,
        transform=train_transform
    )
    val_dataset = ImageFolder(
        root=args.root,
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

    def __call__(self, args, dataloader_name:str):
        return self.dataloader_name_list[dataloader_name](
            args
        )

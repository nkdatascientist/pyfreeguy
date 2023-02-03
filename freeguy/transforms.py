import torch 
from torchvision import transforms
def imagenet_transforms(args, mode):
    if mode == "train":
        return transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomRotation([0, 270]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ])

    if mode == "val":
        return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop([args.image_size, args.image_size]),
            ])


class GetTransforms:
    def __init__(self):
        self.scheduler_name_list = {
            "imagenet": imagenet_transforms
        }

    def __call__(self, args, transforms_name:str, mode:str):
        return self.transforms_name_list[transforms_name](
            args, mode
        )
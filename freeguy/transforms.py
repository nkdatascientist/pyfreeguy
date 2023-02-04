import torch 
from torchvision import transforms

def resnet_imagenet_transforms(args, mode):
    if mode == "train":
        train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomRotation([0, 270]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor()
            ])
        print("Train transforms: \n", train_transforms)
        return train_transforms

    if mode == "val":
        val_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop([args.image_size, args.image_size]),
            ])
        print("Val transforms: \n", val_transforms)
        return val_transforms


class GetTransforms:
    def __init__(self):
        self.transforms_name_list = {
            "resnet_imagenet_transforms": resnet_imagenet_transforms
        }

    def __call__(self, args, mode:str):

        if args.dataloader["transforms"]["name"] in self.transforms_name_list:
            return self.transforms_name_list[args.dataloader["transforms"]["name"]](
                args, mode
            )
        return None

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torch, os

# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
class CocoDataset(Dataset):
    def __init__(self, root, json, transform=None):
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        target = ""
        return image, target

    def __len__(self):
        return len(self.ids)

        

def build_dataloader(args):
    if args.dataset == "COCO2014":
        train_loader, val_loader  = ""
    return train_loader, val_loader

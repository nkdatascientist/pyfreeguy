
from torchvision import transforms
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torch, os

# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
class CocoDataset(Dataset):
    def __init__(self, root, is_train, image_model: torch.nn.Module, txt_model: torch.nn.Module, transform=None):
        self.root = root
        self.coco = COCO(f"{self.root}/annotations/captions_train2014.json")
        self.mode_name = "train"
        if not is_train:
            self.mode_name = "val"
            self.coco = COCO(f"{self.root}/annotations/captions_val2014.json")
        self.ids = list(self.coco.anns.keys())
        self.transform = transform
        self.create_cach_file(image_model, txt_model)
        self.cache_pt = torch.load(f"../../data/{self.mode_name}_coco2014_cache.pt", map_location="cpu")
    
    def __getitem__(self, index, cache=False):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        full_image_path = os.path.join(self.root, path)
        image = Image.open(full_image_path).convert('RGB')

        if cache:
            img = transforms.ToTensor()(img)
            return img, img_id, caption

        return image, target

    def __len__(self):
        return len(self.ids)

    def create_cach_file(self, image_model: torch.nn.Module, txt_model: torch.nn.Module):
        if not os.path.exists(f"../../data/{self.mode_name}_coco2014_cache.pt"):
            cache = {}
            for index in range(len(self.ids)):
                img, img_id, caption = self.__getitem__(index, cache=True)
                image_encoding = image_model(img.unsqueeze())
                text_encoding, token = txt_model(caption)
                if img_id in cache:
                    cache[img_id]["caption"].append(caption)
                    cache[img_id]["token"].append(token)
                    cache[img_id]["text_encoding"].append(text_encoding)
                else:
                    cache[img_id] = {
                        "caption": [caption],
                        "token": [token],
                        "text_encoding": [text_encoding],
                        "image_encoding": image_encoding,
                    }
            torch.save(cache, f"../../data/{self.mode_name}_coco2014_cache.pt")

def build_dataloader(args):
    if args.dataset == "COCO2014":
        train_loader, val_loader  = ""
    return train_loader, val_loader

coco = CocoDataset(
    "/home/nishanth/Desktop/RnD/deep-research-on-ai/data/COCO2014",
    is_train=True
)
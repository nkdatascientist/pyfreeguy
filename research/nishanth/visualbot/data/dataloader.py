
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image
import torch, os

# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
class CocoDataset(Dataset):
    def __init__(self, root, is_train, transform=None, device=None):
        self.root = root
        self.coco = COCO(f"{self.root}/annotations/captions_train2014.json")
        self.mode_name = "train2014"
        if not is_train:
            self.mode_name = "val2014"
            self.coco = COCO(f"{self.root}/annotations/captions_val2014.json")
        self.ids = list(self.coco.anns.keys())
        self.transform = transform
        self.device = device
    
    def __getitem__(self, index, cache=True):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        id_ = coco.getAnnIds(img_id)

        full_image_path = os.path.join(os.path.join(self.root, self.mode_name), path)
        image = Image.open(full_image_path).convert('RGB')
        if cache:
            img = transforms.ToTensor()(image)
            return img, full_image_path, ann_id, img_id, caption

        return image, target

    def __len__(self):
        return len(self.ids)
from memory_profiler import profile

@profile(precision=4)
def create_cach_file(args, train_dataset, image_model: torch.nn.Module, txt_model: torch.nn.Module, device):
    if os.path.exists(os.path.join(f"/media/nishanth/Nishanth M/Dataset/coco2014/train_coco2014_cache.pt")):
        print("Cache file Not found...!!!")
        for batch in tqdm(train_dataset, total=len(train_dataset), desc="Creating cache file: "):
            img, img_path, ann_id, img_id, caption = batch
            fname = f"{ann_id}_{img_id}"
            os.makedirs("/media/nishanth/Nishanth M/Dataset/coco2014/cache/", exist_ok=True)
            text_model_name = args.preparation["model_name"]
            image_model_name = args.image_encoder["model_name"]
            if os.path.exists(os.path.join(f"/media/nishanth/Nishanth M/Dataset/coco2014/cache/{fname}.pt")):
                continue
                # cache = torch.load(os.path.join(f"/media/nishanth/Nishanth M/Dataset/coco2014/cache/{fname}.pt"), map_location="cpu")

            text_encoding, token = txt_model(caption)
            token["input_ids"] = token["input_ids"].cpu().tolist()
            token["attention_mask"] = token["attention_mask"].cpu().tolist()

            image_encoding = image_model(img.unsqueeze(0).to(device))
            cache = {
                "ann_id": ann_id,
                "img_id": img_id,
                "img_path": img_path,
                "caption": caption,
                "token": { 
                    f"{text_model_name}": {
                            "input_ids": token["input_ids"],
                            "attention_mask": token["attention_mask"]
                    }
                },
                "text_encoding": {
                    f"{text_model_name}":{
                        "last_hidden_state" : text_encoding.last_hidden_state.cpu().tolist(),
                        "pooler_output" : text_encoding.pooler_output.cpu().tolist()
                    }
                },
                "image_encoding": { 
                    f"{image_model_name}" : image_encoding.cpu().tolist()},
            }
            torch.save(cache, os.path.join(f"/media/nishanth/Nishanth M/Dataset/coco2014/cache/{fname}.pt"))

def build_dataloader(args, image_model: torch.nn.Module=None, txt_model:torch.nn.Module=None, device=None):
    if args.dataset["dataset_name"] == "COCO2014":
        train_dataset = CocoDataset(
            "/media/nishanth/Nishanth M/Dataset/coco2014/",
            is_train=True,
            device=device
        )
        # from torch.utils.data import Subset
        # train_dataset = Subset(train_dataset, list(range(5000)))
        create_cach_file(args, train_dataset, image_model, txt_model, device)
        exit()
        val_dataset = CocoDataset(
            "/media/nishanth/Nishanth M/Dataset/coco2014/",
            is_train=False,
            image_model=image_model,
            txt_model=txt_model,
            device=device
        )

        train_loader  = DataLoader(
            dataset=train_dataset, 
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
        )
        val_loader  = DataLoader(
            dataset=val_dataset, 
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
        )
    return train_loader, val_loader

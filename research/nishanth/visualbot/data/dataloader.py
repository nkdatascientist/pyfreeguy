
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torch.multiprocessing as mp
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
# from memory_profiler import profile

# @profile(precision=4)
def create_cach_file(args, index, folder, train_dataset, image_model: torch.nn.Module, txt_model: torch.nn.Module, device):

        dataset_path = args.dataset["dataset_path"]
        text_model_name = args.preparation["model_name"]
        image_model_name = args.image_encoder["model_name"]
        os.makedirs(f"{dataset_path}/cache/{folder}", exist_ok=True)
        for batch in tqdm(train_dataset, total=len(train_dataset), desc=f"Creating cache file {index}"):
            img, img_path, ann_id, img_id, caption = batch
            fname = f"{ann_id}_{img_id}"
            
            if os.path.exists(os.path.join(f"{dataset_path}/cache/{folder}/{fname}.pt")):
                continue
                # cache = torch.load(os.path.join(f"{dataset_path}/cache/{fname}.pt"), map_location="cpu")

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
            torch.save(cache, os.path.join(f"{dataset_path}/cache/{folder}/{fname}.pt"))

def main_cache(args, dataset, folder, txt_model, image_model, device):
        num_processes = 4
        data_loader_list = []
        split_limt = list(range(0, len(dataset), int(len(dataset) / num_processes)))
        split_limt[len(split_limt)-1] = len(dataset)
        print("split: ", split_limt)
        for index in range(len(split_limt)-1):
            start = split_limt[index]
            end = split_limt[index + 1]
            sub_dataset = Subset(dataset, list(range(start, end)))
            data_loader_list.append(sub_dataset)

        txt_model.share_memory()
        image_model.share_memory()

        torch.multiprocessing.set_start_method("spawn")
        
        processes = []
        for index in range(num_processes):
            p = mp.Process(target=create_cach_file, args=(args, index, folder, data_loader_list[index], image_model, txt_model, device))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

def build_dataloader(args, bc=False, image_model: torch.nn.Module=None, txt_model:torch.nn.Module=None, device=None):


    if args.dataset["dataset_name"] == "COCO2014":
        train_dataset = CocoDataset(
            args.dataset["dataset_path"],
            is_train=True,
            device=device
        )
        # if bc:
            # main_cache(args, train_dataset, "train", txt_model, image_model, device)
        
        val_dataset = CocoDataset(
            args.dataset["dataset_path"],
            is_train=False,
            device=device
        )
        
        if bc:
           main_cache(args, val_dataset, "val", txt_model, image_model, device)

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

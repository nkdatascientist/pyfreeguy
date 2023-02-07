
import torch
import warnings
warnings.filterwarnings("ignore")

from pytorch_lightning import LightningModule, Trainer
from typing import Any

from freeguy.cerebral.classification import train_model, validation_model
from freeguy.optimizer.optimizer import GetOptimizer
from freeguy.scheduler.scheduler import GetScheduler
from freeguy.dataloader import GetDataloader
from freeguy.models.encoder import Resnet18
from freeguy.utils import AverageMeter, accuracy
from freeguy.loss import GetLoss


class TrainClassificalModel(LightningModule):
    def __init__(self, config: dict, model: torch.nn.Module,
            val_dl: torch.nn.Module, criterion: torch.nn.Module,
            optim: torch.nn.Module, scheduler=None):
        super().__init__()

        self.config = config
        self.model = model
        self.val_dl = val_dl
        self.scheduler = scheduler
        self.criterion = criterion
        self.optim = optim
        
        self.train_loss = AverageMeter("Train_Loss", ":6.6f")
        self.train_top1 = AverageMeter("Top@1", ":6.2f")
        self.train_top5 = AverageMeter("Top@5", ":6.2f")
        
        self.val_loss = AverageMeter("Val_Loss", ":6.6f")
        self.val_top1 = AverageMeter("Top@1", ":6.2f")
        self.val_top5 = AverageMeter("Top@5", ":6.2f")
    
    def forward(self, img: torch.nn.Module):
        return self.model(img)
    
    def configure_optimizers(self):
        return self.optim

    def training_step(self, train_batch, train_batch_idx):
        img, labels = train_batch 
        logits = self(img)
        train_loss = self.criterion(logits, labels)
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        
        self.train_top1.update(acc1[0].item(), labels.size(0))
        self.train_top5.update(acc5[0].item(), labels.size(0))
        self.train_loss.update(train_loss.item(), labels.size(0))
        
        self.log('train_loss', train_loss.item(), on_step=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, val_batch, val_batch_idx):
        img, labels = val_batch 
        logits = self(img)
        vloss = self.criterion(logits, labels)
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        
        self.val_top1.update(acc1[0].item(), labels.size(0))
        self.val_top5.update(acc5[0].item(), labels.size(0))
        self.val_loss.update(vloss.item(), labels.size(0))
        
        self.log('val_loss',vloss.item(), on_step=True, prog_bar=True, logger=True)
        return vloss

    def validation_epoch_end(self, outputs):
        self.log('val/top1', self.val_top1.avg)
        self.log('val/top5', self.val_top5.avg)
        self.log('val/loss', self.val_loss.avg)
        

    def training_epoch_end(self, outputs):
        self.log('train/top1', self.val_top1.avg)
        self.log('train/top5', self.val_top5.avg)
        self.log('train/loss', self.val_loss.avg)

def main(args):

    model = Resnet18()
    # from torchvision.models import resnet18
    # model = resnet18(pretrained=True)
    optim = GetOptimizer()(args, model)
    criterion =  GetLoss()(args)
    scheduler = GetScheduler()(args, optim)
    train_dl, val_dl = GetDataloader()(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.train:
        from pytorch_lightning.profilers import PyTorchProfiler

        profiler = PyTorchProfiler()
        model = TrainClassificalModel(
            config = args,
            model = model,
            val_dl = val_dl,
            scheduler = scheduler,
            criterion = criterion,
            optim = optim
        )

        # https://pytorch-lightning.readthedocs.io/en/stable/tuning/profiler_intermediate.html
        worker = Trainer(
            profiler=profiler,
            accelerator="gpu",
            devices=1,
            precision=16
        )
        worker.fit(
            model, 
            train_dataloaders=train_dl, 
            val_dataloaders=val_dl
        )

    else:
        validation_model(
            model, val_dl, criterion, device
        )

if __name__ == "__main__":
    from argparse import Namespace
    import json, argparse

    parser = argparse.ArgumentParser('classification fine-tuning')
    parser.add_argument('--config', default=None, type=str, help='config path')
    parser.add_argument('--train', action='store_true', help='train model', default=False)
    parser.add_argument('--val',   action='store_true', help='validate model', default=False)
    args = parser.parse_args()

    temp_args = vars(args)
    with open(args.config, "r") as f:
        data = json.load(f)

    for item in data:
        temp_args[item] = data[item]

    # https://stackoverflow.com/questions/2597278/python-load-variables-in-a-dict-into-namespace
    temp_args = Namespace(**temp_args)
    
    main(args)
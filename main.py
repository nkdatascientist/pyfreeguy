
"""
# 1 - fp32 pipeline setup 
#       1. Fp16 
#       2. 
# 2 - clip training
# 3 - fc -> conv
# 4 - new scheduler 
# 5 - innovation
"""

from freeguy.cerebral.classification import train_model, validation_model
from freeguy.cerebral.tensorboard import GetSummaryWriter
from freeguy.cerebral.scheduler import GetScheduler
from freeguy.cerebral.optimizer import GetOptimizer
from freeguy.dataloader import GetDataloader
from freeguy.models.encoder import Resnet18
from freeguy.cerebral.loss import GetLoss
from argparse import Namespace
import json, argparse
import torch

def main(args):

    model = Resnet18()
    criterion = GetLoss(args)
    optimizer = GetOptimizer()(args, model)
    scheduler = GetScheduler()(optimizer)
    summarywriter = GetSummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader = GetDataloader(args)

    epochs = epochs + 1
    for epoch in range(1, epochs):
        train_loss, train_top1, train_top5 = train_model(
            epoch, model, train_dataloader, criterion, device, optimizer, summarywriter
        )
        # Training Loss
        summarywriter.insert("/training/top1", train_top1, epoch)
        summarywriter.insert("/training/top5", train_top5, epoch)
        summarywriter.insert("/training/loss", train_loss, epoch)

        val_loss, val_top1, val_top5 = validation_model(
            epoch, model, val_dataloader, criterion, device
        )
        # Validation Loss
        summarywriter.insert("/validation/top1", val_top1, epoch)
        summarywriter.insert("/validation/top5", val_top5, epoch)
        summarywriter.insert("/validation/loss", val_loss, epoch)
        
        scheduler.step()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Resnet fine-tuning')
    parser.add_argument('--config', default=None, type=str, help='config path')
    args = parser.parse_args()

    temp_args = vars(args)
    with open(args.config, "r") as f:
        data = json.load(f)

    for item in data:
        temp_args[item] = data[item]

    # https://stackoverflow.com/questions/2597278/python-load-variables-in-a-dict-into-namespace
    temp_args = Namespace(**args)
    main(args)
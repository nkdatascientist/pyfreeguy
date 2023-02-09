
"""
# !-TODO
# 1 - fp32 pipeline setup
#     1. Fp16
#     2. Quantization
#          - Int4A4 or Int4A8
#          - FP16
# 2 - clip training
# 3 - fc -> conv
# 4 - new scheduler
# 5 - innovation
"""

from freeguy.cerebral.classification import train_model, validation_model
from freeguy.tensorboard import GetSummaryWriter
from freeguy.scheduler.scheduler import GetScheduler
from freeguy.optimizer.optimizer import GetOptimizer
from freeguy.dataloader import GetDataloader
from freeguy.models.encoder import Resnet18
from freeguy.loss import GetLoss
from argparse import Namespace
import json, argparse, torch


def train(model, train_dataloader, val_dataloader, criterion, device, optimizer,
         summarywriter: GetSummaryWriter, scheduler, args):

    args.epochs = args.epochs + 1
    for epoch in range(1, args.epochs):
        train_loss, train_top1, train_top5 = train_model(
            epoch, model, train_dataloader, criterion, device, optimizer, summarywriter
        )

        # Training Loss
        summarywriter.insert("training/top1", train_top1, epoch)
        summarywriter.insert("training/top5", train_top5, epoch)
        summarywriter.insert("training/avg_loss", train_loss, epoch)

        val_loss, val_top1, val_top5 = validation_model(
            model, val_dataloader, criterion, device 
        )

        # Validation Loss
        summarywriter.insert("validation/top1", val_top1, epoch)
        summarywriter.insert("validation/top5", val_top5, epoch)
        summarywriter.insert("validation/loss", val_loss, epoch)
        
        if args.scheduler["name"] == "ReduceLROnPlateau":
            scheduler.step(f"{train_loss:.5f}")
        else:
            scheduler.step()
        summarywriter.export()
    
def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    model = Resnet18().to(device)
    print("Model Build Successfully")
    
    criterion = GetLoss()(args)
    print("Loaded Loss Function Successfully")

    train_dataloader, val_dataloader = GetDataloader()(args)
    print("Loaded Data Loader Successfully")
    
    if args.train_model:
        optimizer = GetOptimizer()(args, model)
        print("Loaded Optimizer Successfully")

        scheduler = GetScheduler()(args, optimizer)
        print("Loaded Scheduler Successfully")

        summarywriter = GetSummaryWriter(args)
        print("Loaded Summary Writer Successfully")

        train(
            model, train_dataloader, val_dataloader, criterion, device, optimizer, summarywriter, scheduler, args
        )

    if args.val_model:
        validation_model(
            model, val_dataloader, criterion, device
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser('classification fine-tuning')
    parser.add_argument('--config', default=None, type=str, help='config path')
    parser.add_argument('--train_model', action='store_true', help='train model', default=False)
    parser.add_argument('--val_model',   action='store_true', help='validate model', default=False)

    args = parser.parse_args()

    temp_args = vars(args)
    with open(args.config, "r") as f:
        data = json.load(f)

    for item in data:
        temp_args[item] = data[item]

    # https://stackoverflow.com/questions/2597278/python-load-variables-in-a-dict-into-namespace
    temp_args = Namespace(**temp_args)
    main(args)
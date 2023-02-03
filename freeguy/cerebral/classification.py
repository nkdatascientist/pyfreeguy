import torch
from freeguy.utils import AverageMeter, accuracy
from tqdm import tqdm 

def train_model(epoch: int, model: torch.nn.Module, train_dataloader, criterion, device,
    optimizer, summarywriter):
    model.train()
    train_loss = AverageMeter("Train_Loss", ":6.6f")
    top1 = AverageMeter("Top@1", ":6.2f")
    top5 = AverageMeter("Top@5", ":6.2f")

    running_loss = AverageMeter("Running_Loss", ":6.6f")
    for index, (images, targets) in enumerate(tqdm(len(train_dataloader), desc=f"Epoch: {epoch}")):
        images, targets = images.to(device), targets.to(device)
        
        output = model(images)
        loss = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        
        top1.update(acc1[0].item(), targets.size(0))
        top5.update(acc5[0].item(), targets.size(0))
        train_loss.update(loss.item(), targets.size(0))
        running_loss.update(loss.item(), targets.size(0))
        if index % 100 == 99:
            summarywriter.insert("/training/top1", loss.item(), epoch * index)
            running_loss.reset()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss.avg, top1.avg, top5.avg

# https://pytorch.org/docs/stable/generated/torch.no_grad.html
@torch.no_grad()
def validation_model(epoch: int, model: torch.nn.Module, val_dataloader, criterion, device):
    
    model.eval()
    val_loss = AverageMeter("Val_Loss", ":6.6f")
    top1 = AverageMeter("Top@1", ":6.2f")
    top5 = AverageMeter("Top@5", ":6.2f")
    for index, (images, targets) in enumerate(tqdm(len(val_dataloader), desc=f"Epoch: {epoch}")):
        images, targets = images.to(device), targets.to(device)
        
        output = model(images)
        loss = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        
        top1.update(acc1[0].item(), targets.size(0))
        top5.update(acc5[0].item(), targets.size(0))
        val_loss.update(loss.item(), targets.size(0))

    return val_loss.avg, top1.avg, top5.avg
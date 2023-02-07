import torch
from freeguy.utils import AverageMeter, accuracy
from tqdm import tqdm 

def train_model(epoch: int, model: torch.nn.Module, train_dataloader, criterion, device,
    optimizer, summarywriter):
    model.train()
    train_loss = AverageMeter("Train_Loss", ":6.6f")
    top1 = AverageMeter("Top@1", ":6.2f")
    top5 = AverageMeter("Top@5", ":6.2f")

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    running_loss = AverageMeter("Running_Loss", ":6.6f")
    for index, (images, targets) in enumerate(train_dataloader):
        images, targets = images.to(device), targets.to(device)
        
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        
        top1.update(acc1[0].item(), targets.size(0))
        top5.update(acc5[0].item(), targets.size(0))
        train_loss.update(loss.item(), targets.size(0))
        running_loss.update(loss.item(), targets.size(0))

        if index % 100 == 99:
            lr = get_lr(optimizer)
            weight_decay = optimizer.defaults['weight_decay']
            print(f"Epoch: {epoch} Iteration: [{index}/{len(train_dataloader)}] Top1: {top1.avg} Top5: {top5.avg} LR: {lr} Loss: {running_loss.avg}")
            
            summarywriter.insert("training/loss", running_loss.avg, epoch * index)
            summarywriter.insert("optimizer/learning_rate", lr, epoch * index)
            summarywriter.insert("optimizer/weight_decay", weight_decay, epoch * index)
            running_loss.reset()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss.avg, top1.avg, top5.avg

# https://pytorch.org/docs/stable/generated/torch.no_grad.html
@torch.no_grad()
def validation_model(model: torch.nn.Module, val_dataloader, criterion, device, epoch=None):
    
    model.eval()
    top1 = AverageMeter("Top@1", ":6.2f")
    top5 = AverageMeter("Top@5", ":6.2f")
    val_loss = AverageMeter("Val_Loss", ":6.6f")

    description = f"Epoch: {epoch}" if epoch else "Validation "
    for index, (images, targets) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=description):
        images, targets = images.to(device), targets.to(device)
        
        output = model(images)
        loss = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        
        top1.update(acc1[0].item(), targets.size(0))
        top5.update(acc5[0].item(), targets.size(0))
        val_loss.update(loss.item(), targets.size(0))

    print("Top 1: ", top1.avg)
    print("Top 5: ", top5.avg)
    print("Val Loss: ", val_loss.avg)
    return val_loss.avg, top1.avg, top5.avg
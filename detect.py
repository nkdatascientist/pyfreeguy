
from freeguy.cerebral.detection import train_model, val_model
from freeguy.optimizer.optimizer import GetOptimizer
from freeguy.checkpoints import saveckpt, loadckpt
from freeguy.tensorboard import GetSummaryWriter
from freeguy.dataloader import GetDataloader
from freeguy.models.detection import YOLOv3
from freeguy.loss import GetLoss
import torch

def train(args, model, train_dataloader, test_dataloader, optimizer,
            loss_fn, device):

    scaler = torch.cuda.amp.GradScaler()
    scaled_anchors = (
        torch.tensor(args.anchors)
        * torch.tensor(args.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    summarywriter = GetSummaryWriter(args)
    print("Loaded Summary Writer Successfully")

    args.epochs = args.epochs + 1
    for epoch in range(1, args.epochs):

        train_model(
            train_dataloader, model, optimizer, loss_fn, scaler, scaled_anchors, device, summarywriter, epoch
        )

        mapval = val_model(
            args, model, test_dataloader, device
        )        
        # Validation Loss
        summarywriter.insert("validation/top1", mapval.item(), epoch)

        if args.save_ckpt:
           saveckpt(model, optimizer, filename=f"checkpoint.pth.tar")
        print(f"MAP: {mapval.item()}")

def main(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_dataloader, test_dataloader, val_dataloader = GetDataloader()(args)

    model = YOLOv3(num_classes=args.num_classes).to(device)
    # optimizer = GetOptimizer()(args, model)
    loss_fn = GetLoss()(args)

    # if args.ckpt:
    #     loadckpt(
    #         args.CHECKPOINT_FILE, model, optimizer, args.LEARNING_RATE
    #     )

    if args.train_model:
        optimizer = GetOptimizer()(args, model)
        print("Loaded Optimizer Successfully")

        train(
            args, model, train_dataloader, test_dataloader, optimizer,
                loss_fn, device
        )

    if args.val_model:
        val_model(
            args, model, test_dataloader, device
        )


if __name__ == "__main__":

    from argparse import Namespace
    import json, argparse

    parser = argparse.ArgumentParser('classification fine-tuning')
    parser.add_argument('--config', default=None, type=str, help='config path')
    parser.add_argument('--train_model', action='store_true', help='train model', default=False)
    parser.add_argument('--val_model',   action='store_true', help='validate model', default=False)
    args = parser.parse_args()

    temp_args = vars(args)
    with open(args.config, "r") as f:
        data = json.load(f)

    data["S"] = [data["image_size"] // 32, data["image_size"] // 16, data["image_size"] // 8]
    data["anchors"] = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ] 
    for item in data:
        temp_args[item] = data[item]

    # https://stackoverflow.com/questions/2597278/python-load-variables-in-a-dict-into-namespace
    temp_args = Namespace(**temp_args)
    main(args)
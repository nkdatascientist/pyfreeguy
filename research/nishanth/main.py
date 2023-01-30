
from visualbot.models import TextEncoder, build_model
from visualbot.utils import Dataset_Preparation
from visualbot.data import build_dataloader

from transformers import GPT2Model
from torch import optim
import argparse, os
import json, torch 


# https://towardsdatascience.com/train-and-deploy-fine-tuned-gpt-2-model-using-pytorch-on-amazon-sagemaker-to-classify-news-articles-612f9957c7b
# https://github.com/haocai1992/GPT2-News-Classifier/blob/main/sagemaker-train-deploy/code/train_deploy.py
# https://www.kaggle.com/code/baekseungyun/gpt-2-with-huggingface-pytorch
# https://github.com/hyunwoongko/transformer
def train_one_epoch(model, train_dataloader, optimizer, criterion, device):

    # https://www.kaggle.com/code/baekseungyun/gpt-2-with-huggingface-pytorch
    for data, labels in train_dataloader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

def validation():
    pass 

def train(model, train_dataloader, val_dataloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(2):
        model.train()
        train_acc, train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        val_acc, val_loss = validation()

def sample_test():
    img = torch.randn(1,3,224,224)
    text = "my name is nishanth"

    img_encoder = ImageEncoder()
    text_encoder = TextEncoder()

    print("#"*20)
    print("model build completed...!!!")
    print("#"*20)

    print("Image encoder", img_encoder(img).shape)
    text_output, token = text_encoder(text)
    for k, v in token.items():
        print(k, v)
        print(k, v.shape)
    print("text encoder", text_output.last_hidden_state.shape)
    print("Text encoder", text_output.pooler_output.shape)

    exit()

def background_check(args, device):

    # Download dataset
    Dataset_Preparation(args)()
    print("===> Dataset Preparation Completed...!!!")
    

def main(args, device):

    if args.sample_test:
        sample_test() 

    background_check(args, device)
    # convert dataset to cache file
    txt_model = TextEncoder(args)
    train_dataloader, val_dataloader = build_dataloader(args, txt_model)
    del txt_model

    model = build_model(args)
    if args.training:
        train(model, train_dataloader, val_dataloader, device)

if __name__ == "__main__":

    def helper(args):
        with open(args.config, "r") as f:
            args = json.load(f)
        return argparse.Namespace(**args)

    parser = argparse.ArgumentParser()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    parser.add_argument('--config',           default="./config/baseline.json",             help='Config Path')    
    args = parser.parse_args()

    args = helper(args)
    main(args, device)
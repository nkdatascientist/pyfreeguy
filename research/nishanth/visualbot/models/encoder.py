import torch
from torchvision.models import resnet18
# /home/nishanth/anaconda3/envs/aimet_1.23/lib/python3.8/site-packages/transformers/models/roberta/modeling_roberta.py
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModel

# https://huggingface.co/openai/clip-vit-large-patch14
class ImageEncoder(torch.nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        if args.model == "resnet18":
            self.model = resnet18(pretrained=True, progress=True)

    def forward(self, img: torch.Tensor):
        return self.model(img)

class TextEncoder(torch.nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.args = args
        self.model_name = self.args.generative_retrival["model_name"]
        if self.model_name in ["roberta-large-openai-detector"]:
            self.text_tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            self.text_model = RobertaModel.from_pretrained(self.model_name)
        elif self.model_name in ["bert-base-uncased"]:
            # https://huggingface.co/bert-base-uncased
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.text_model = AutoModel.from_pretrained(self.model_name)
        else:
            assert False, f"Model {self.model_name} is Not supported"
    
    def forward(self, txt: str):
        token = self.text_tokenizer(txt, return_tensors='pt')
        return self.text_model(**token), token
        
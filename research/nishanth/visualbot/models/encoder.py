import torch
from torchvision.models import resnet18
# /home/nishanth/anaconda3/envs/aimet_1.23/lib/python3.8/site-packages/transformers/models/roberta/modeling_roberta.py
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModel

# https://huggingface.co/openai/clip-vit-large-patch14
class ImageEncoder(torch.nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        if args.image_encoder["model_name"] == "resnet18":
            self.model = resnet18(pretrained=True, progress=True)

    def forward(self, img: torch.Tensor):
        return self.model(img)

class TextEncoder(torch.nn.Module):
    def __init__(self, args, device):
        super(TextEncoder, self).__init__()
        self.args = args
        self.device = device
        self.model_name = self.args.preparation["model_name"]
        if self.model_name in ["roberta-base-openai-detector", "roberta-large-openai-detector"]:
            self.text_tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            self.text_model = RobertaModel.from_pretrained(self.model_name)
        elif self.model_name in ["bert-base-uncased", "bert-large-uncased"]:
            # https://huggingface.co/bert-base-uncased
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.text_model = AutoModel.from_pretrained(self.model_name)
        else:
            assert False, f"Model {self.model_name} is Not supported"
        self.text_model = self.text_model.to(self.device)
    
    # https://discuss.huggingface.co/t/is-transformers-using-gpu-by-default/8500/2
    def forward(self, txt: str):
        token = self.text_tokenizer(txt, return_tensors='pt').to(self.device)
        return self.text_model(**token), token

        
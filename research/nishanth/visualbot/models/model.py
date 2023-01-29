
import torch
from visualbot.models import ImageEncoder, GenerativeRetrival

class VisualBot(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args  = args

    def __call__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, img: torch.Tensor, txt, annotation=None):
        if not self.encoder:
            output = self.decoder(img, txt, annotation)
            return output
        if not self.decoder:
            logits = self.encoder(img)
            return logits
        logits = self.encoder(img)
        output = self.decoder(logits, txt, annotation)
        return output

def build_model(args):
    if args.image_encoder:
        backbone = ImageEncoder(args)
    if args.generative_retrival:
        generative_retrival = GenerativeRetrival(args)
    return VisualBot(args)(backbone, generative_retrival)


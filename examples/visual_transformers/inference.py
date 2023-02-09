
# !wget "https://raw.githubusercontent.com/jankrepl/mildlyoverfitted/master/github_adventures/vision_transformer/classes.txt"
# !wget "https://raw.githubusercontent.com/jankrepl/mildlyoverfitted/master/github_adventures/vision_transformer/cat.png"

import numpy as np
from PIL import Image
import torch

# load label .txt file
imagenet_labels = dict(enumerate(open("classes.txt")))

model = torch.load("visual_tranformer.pth")
model.eval()

img = (np.array(Image.open("cat.png")) / 128) - 1  # in the range -1, 1
inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

logits = model(inp)
probs = torch.nn.functional.softmax(logits, dim=-1)

top_k = 10
top_probs, top_ixs = probs[0].topk(top_k)
for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
    ix = ix_.item()
    prob = prob_.item()
    cls = imagenet_labels[ix].strip()
    print(f"{i}: {cls:<45} --- {prob:.4f}")
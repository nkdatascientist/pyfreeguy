"""
@author : Hyunwoong
@when : 2019-12-19
@homepage : https://github.com/gusdnd852
"""

import math
from collections import Counter

import numpy as np

from data import *
from models.model.transformer import Transformer
# from util.bleu import get_bleu, idx_to_word


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=0.00,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')

def main_process():
    src = torch.rand((1, 14), device=device).to(torch.int64)
    trt = torch.rand((1, 11), device=device).to(torch.int64)
    print(model(src, trt).shape)

if __name__ == '__main__':
    main_process()

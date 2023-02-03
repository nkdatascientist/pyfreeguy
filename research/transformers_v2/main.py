# import torch
# from vit_pytorch import ViT
# from torchinfo import summary

# v = ViT(
#     image_size = 256,
#     patch_size = 32,
#     num_classes = 1000,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

# img = torch.randn(1, 3, 256, 256)
# # summary(v, (1,3,256,256))

# preds = v(img) # (1, 1000)
# print(preds.shape)

# """
# ====================================================================================================
# Layer (type:depth-idx)                             Output Shape              Param #
# ====================================================================================================
# ViT                                                [1, 1000]                 67,584
# ├─Sequential: 1-1                                  [1, 64, 1024]             --
# │    └─Rearrange: 2-1                              [1, 64, 3072]             --
# │    └─Linear: 2-2                                 [1, 64, 1024]             3,146,752
# ├─Dropout: 1-2                                     [1, 65, 1024]             --
# ├─Transformer: 1-3                                 [1, 65, 1024]             --
# │    └─ModuleList: 2-3                             --                        --
# │    │    └─ModuleList: 3-1                        --                        8,396,800
# │    │    └─ModuleList: 3-2                        --                        8,396,800
# │    │    └─ModuleList: 3-3                        --                        8,396,800
# │    │    └─ModuleList: 3-4                        --                        8,396,800
# │    │    └─ModuleList: 3-5                        --                        8,396,800
# │    │    └─ModuleList: 3-6                        --                        8,396,800
# ├─Identity: 1-4                                    [1, 1024]                 --
# ├─Sequential: 1-5                                  [1, 1000]                 --
# │    └─LayerNorm: 2-4                              [1, 1024]                 2,048
# │    └─Linear: 2-5                                 [1, 1000]                 1,025,000
# ====================================================================================================
# Total params: 54,622,184
# Trainable params: 54,622,184
# Non-trainable params: 0
# Total mult-adds (M): 54.55
# ====================================================================================================
# Input size (MB): 0.79
# Forward/backward pass size (MB): 29.29
# Params size (MB): 218.22
# Estimated Total Size (MB): 248.30
# ====================================================================================================
# """

# Single Input for FC layer
import torch
inputs = torch.tensor([
    # [1., 2., 3.],
    [10., 21., 343.],
])
in_channel = 3
out_channel = 6
lin_fc = torch.nn.Linear(in_channel, out_channel)
conv = torch.nn.Conv1d(in_channel, out_channel, kernel_size=1)

# print(lin_fc.weight.data.shape)
# print(lin_fc.bias.data.shape)
# print(conv.weight.data.shape)
# print(conv.bias.data.shape)

lin_fc.weight.data = torch.ones(out_channel,in_channel)
lin_fc.bias.data = torch.ones(out_channel)
conv.weight.data = torch.ones(out_channel,in_channel,1)
conv.bias.data = torch.ones(out_channel)

print("Input:",inputs)
print("Input shape:",inputs.shape)
print("FC Out: ",lin_fc(inputs))
print("Conv Out: ",conv(inputs.view(3,1)))

# input = torch.rand(1, 2048, 65)
# conv = torch.nn.Conv1d(2048, 2048, kernel_size=1)
# print(conv(inputs).permute(0, 2, 1).shape)
exit()

# Multile Input for FC layer
import torch
inputs = torch.tensor([[
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.],
]])
in_channel = 3
out_channel = 6
lin_fc = torch.nn.Linear(in_channel, out_channel)
conv = torch.nn.Conv1d(in_channel, out_channel, kernel_size=1)

# print(lin_fc.weight.data.shape)
# print(lin_fc.bias.data.shape)
# print(conv.weight.data.shape)
# print(conv.bias.data.shape)

lin_fc.weight.data = torch.ones(out_channel,in_channel)
lin_fc.bias.data = torch.ones(out_channel)
conv.weight.data = torch.ones(out_channel,in_channel,1)
conv.bias.data = torch.ones(out_channel)

print("Input:",inputs)
print("Input shape:",inputs.shape)
print("FC Out: ",lin_fc(inputs))
print("Conv Out: ",conv(inputs))

# input = torch.rand(1, 2048, 65)
# conv = torch.nn.Conv1d(2048, 2048, kernel_size=1)
# print(conv(inputs).permute(0, 2, 1).shape)
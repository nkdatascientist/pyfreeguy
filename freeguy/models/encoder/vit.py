"""

Model: Visual Transformers(VIT)
Invented by Google Research

Notes:
Image is converted to patch 





# * Dropout Layer
import torch 
module = torch.nn.Dropout(0.5)
inp = torch.ones(3,3)
midule(inp) # ! the values will keep on changing for every time
>>> tensor([[1,0,0],
            [0,2,1],
            [2,0,0]])

# * Fully Connected Layer
>>> import torch
>>> fc = torch.nn.Linear(10, 20)
>>> inp2d = torch.nn.randn(40, 10)
>>> fc(input2d).shape
>>> torch.Size([40, 20])

>>> inp3d = torch.nn.randn(40, 33, 10)
>>> fc(input3d).shape
>>> torch.Size([40, 33, 20])

>>> inp5d = torch.nn.randn(40, 2, 3, 4, 10)
>>> fc(input5d).shape
>>> torch.Size([40, 2, 3, 4, 20])

>>> inp7d = torch.nn.randn(40, 2, 3, 4, 5, 6, 10)
>>> fc(input7d).shape
>>> torch.Size([40, 2, 3, 4, 5, 6, 20])

# * Layer Normalization
>>> import torch
>>> inp = torch.tensor([[0.,4.], [-1,7], [3,5]])
>>> n_samples, n_feature = inp.shape
>>> module = torch.nn.LayerNorm(n_feature, elementwise_affine=False) # ! elementwise_affine is same as the bias
>>> sum(p.numel() for p in module.parameters() id p.requirems_grad)
>>> 0
>>> inp.mean(-1), inp.std(-1, unbaised=False)
>>> (torch.tensor([2.,3.,4.]), torch.tensor([2.,4.,1.]))

"""

import torch 

class PatchEmbed(torch.nn.Module):
    """
    Split image into patches and embed them
    
    * Parameters
    img_size  : int  size of the image
    patch_size: int  Number of patch
    in_channel: int  Number of channels
    embed_dim : int  Number od dimensions # ! Note: embeddings will stays same all over the model
    
    * Attributes
    n_patches: int
    proj: nn.conv2d 
            1. we take kernel and stride size equal to number of patches
               so that we can cover image or ignore overlap
            2. kernel size will falls under the patch size  
    """
    def __init__(self, img_size:int, patch_size:int, in_channel:int, embed_dim=768):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        # ? this formula will cover all the area of the image
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = torch.nn.Conv2d(
            in_channel,
            embed_dim,
            patch_size, # ! kernel 
            patch_size  # ! stride 
        )

    def forward(self, x: torch.Tensor ):
        """
        x: torch.Tensor => # B, C, H, W

        """
        x = self.proj(x)   # ! 4D tensor 
        x = x.flattern(2)    # ! we will take last two dimensions -> single dimension
        x = x.transpose(1,2) 
        return x

class Attention(torch.nn.Module):
    """
    # * https://data-science-blog.com/wp-content/uploads/2022/01/mha_img_original.png
    Attention Mechanism

    * Parameter 
    dim         : int    Number of input output dimension
    n_heads     : int    Numbers of attention heads
    qkv_bias    : bool   Include bias to query, key & value projections
                * if True   => The parameter of the model will change
                * if False  => The parameter of the model remains same
    attn_p      : float  Dropout Ration(0 to q) applied to query, key & value tensors
    proj_p      : float  Dropout Ration(0 to q) applied to output tensors

    * Attributes
    scale                   : float         Normalizaing consant for the dot ptoduct 
    qkv                     : nn.Linear     Query, key and value
    proj                    : nn.Liner      Linear mapping that takes in the concatenate 
                                            the output of all the attention head and map 
                                            it into a new space
    attn_drop, proj_drop    : nn.Dropout    Dropout Layer
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        # * Reason : Set it up in this way is that once we concatenate all the attention heads 
        # *          we'll get a new tensor that will have the same dim as the input   
        self.head_dim = dim // n_heads #  define dim for each and every head 
        self.scale = self.head_dim ** -0.5 # Not to feed extreamely large values

        self.qkv = torch.nn.Linear(dim, dim*3, bias=qkv_bias)
        self.qkv_drop = torch.nn.Dropout(attn_p)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_p)

    def forward(self, x):
        """
        # ! this function input and output will have same dimensions 
        x : torch.Tensor  => (sample, npatch + 1, dim) 
                          => one is that we will always have the class token as the first token in hte sequence   
        """
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(x) # (n_s, n_p + 1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )   # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1,4
        )   #  3, n_samples, n_heads, n_patches + 1, head_dim
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2,-1)
        
        dp = ( q @ k_t ) * self.scale 
        attn = dp.softmax(dim=-1)  # we are doing the weighted average for descrite weighted average 
        
        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(1,2)
        weighted_avg = weighted_avg.flatten(2)

        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x

class MLP(torch.nn.Module):
    """
    Multilayer Perceptron

    * Parameter
    in_feature  : int
    hid_feature : int
    out_feature : int
    drop_out    : int

    * Activation funtion -> Gelu
    """
    def __init__(self, in_features, hidden_feature, out_features, drop_ratio=0.):
        super().__init__()

        self.fc1 = torch.nn.Linear(in_features, hidden_feature)
        self.act = torch.nn.GELU() 
        self.fc2 = torch.nn.Linear(hidden_feature, out_features)
        self.drop = torch.nn.Dropout(drop_ratio)

    def forward(self, x):   
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(torch.nn.Module):
    """
    Combining all to gether

    dim         : int
    n_heads     : int
    ml_ration   : float  # ? input for hidden feature
    qkv_ratio   : bool
    p, attn_p   : float  # ? Dropout ratio 
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, proj_p=0., attn_p=0.):
        super().__init__()

        self.norm1 = torch.nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = torch.nn.LayerNorm(dim, eps=1e-6)
        self.atten = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=proj_p
        )
        self.mlp = MLP(
            in_features=dim
            hidden_feature= int(dim * mlp_ratio)
            out_features=dim
        )

    # This block will follow the same dimension
    def forward(self, x):
        x = x + self.atten(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x 


class VisualTransoormer(torch.nn.Module):
    """ 
    Simplified Version of Vision Transofrmer
    
    # * Parameter
    img_size : int
    patch_size: int
    in_chans: int
    n_classes: int
    embed_dim: int
    depth: int          # Depth / Number of the blocks  
    n_heads: int
    mlp_ratio: float
    qkv_ratio: bool
    p, attn_p: float    # Dropout ratio
    
    # * Attribute
    patch_embed :
    cls_token   :
    pos_emb     :
    pos_drop    :
    blocks      :
    norm        :
    
    """
    def __init__(self, 
        img_size=384,
        patch_size=16,
        n_classes=1000,
        in_channels=3,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4,
        qkv_bias=True, 
        p=0., 
        atnn_p=0.
    ):
        super().__init__()


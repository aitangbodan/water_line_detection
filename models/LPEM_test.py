
import torch
import torch.nn as nn

from time import time

from timm.models.layers import DropPath

# ResMLP's normalization
class Aff(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # learnable
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x

# Color Normalization
class Aff_channel(nn.Module):
    def __init__(self, dim, channel_first = True):
        super().__init__()
        # learnable
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))
        self.color = nn.Parameter(torch.eye(dim))
        self.channel_first = channel_first

    def forward(self, x):
        if self.channel_first:
            x1 = torch.tensordot(x, self.color, dims=[[-1], [-1]])
            x2 = x1 * self.alpha + self.beta
        else:
            x1 = x * self.alpha + self.beta
            x2 = torch.tensordot(x1, self.color, dims=[[-1], [-1]])
        return x2

class Mlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LPEM(nn.Module):
    def __init__(self, dim, mlp_ratio=4., k=3, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=Aff_channel, init_values=1e-4):
        super().__init__()
        self.k = k
        self.res1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        #self.norm1 = Aff_channel(dim)
        self.norm1 = norm_layer(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, self.k, padding=1, groups=dim, dilation=1)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.norm2 = Aff_channel(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.beta = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.res1(x)
        B, C, H, W = x.shape
        #print(x.shape)
        norm_x = x.flatten(2).transpose(1, 2)
        #print(norm_x.shape)
        norm_x = self.norm1(norm_x)
        norm_x = norm_x.view(B, H, W, C).permute(0, 3, 1, 2)


        x = x + self.drop_path(self.beta*self.conv2(self.attn(self.conv1(norm_x))))
        return x


if __name__ == "__main__":
    LPEM_block = LPEM(dim = 128)
    x = torch.Tensor(1, 128, 40, 40)
    t1 = time()
    x = LPEM_block(x)
    print(x.shape, time() - t1)

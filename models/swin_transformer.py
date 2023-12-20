# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from audioop import cross
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .senet import se_resnext50_32x4d
import numpy as np
import torch.nn.functional as F


# B, C, H, W; mean var on HW
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size() # B L C
        # print(x.shape)
        # x = x.permute(0,2,1)
        y = self.avg_pool(x).view(b, c)#先pooling获取通道维度，也就是把H，w=>1
        # print(y.shape)# 4,1024
        y = self.fc(y)
        # print(y.shape)
        y = y.view(b, c, 1)#全连接层对通道系数进行调整
        #tensor_1.expand_as(tensor_2) ：把tensor_1扩展成和tensor_2一样的形状
        # y = y.permute(0,2,1)
        return x * y.expand_as(x)#原值乘以对应系数

    def forward_o(self, x):
        b, c, _, _ = x.size() 
        y = self.avg_pool(x).view(b, c)#先pooling获取通道维度，也就是把H，w=>1
        y = self.fc(y).view(b, c, 1, 1)#全连接层对通道系数进行调整
        #tensor_1.expand_as(tensor_2) ：把tensor_1扩展成和tensor_2一样的形状
        return x * y.expand_as(x)#原值乘以对应系数


class Mlp(nn.Module):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape #4,56,56,128
    # print(x.shape)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)#4,8,7,8,7,128
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C) #4,8,8,7,7,128 -> 4*8*8,7,7,128
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    # 4*8*8,7,7,128
    # 7
    # 56,56
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # 4*8*8/8*8=4
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # 4,8,8,7,7,128
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    # 4,8,7,8,7,128 -> 4,56,56,128
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH 13*13,4

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])# [0,1,2,3,4,5,6]
        coords_w = torch.arange(self.window_size[1])# [0,1,2,3,4,5,6]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww 49*49
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.qkv_band = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    # def f2(self, x, mask=None):
    #     B, N, C = x.shape # 4*8*8,7*7,128
    #     qkv = self.qkv_band(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    #     qkv = qkv.permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
    #     # 4*8*8, 4, 7*7, 128/4
    #     q = q.transpose(-2, -1)
    #     # 4*8*8, 4, 32, 49
    #     k = k.transpose(-2, -1)
    #     v = v.transpose(-2, -1)

    #     q = torch.nn.functional.normalize(q, dim=-1)
    #     k = torch.nn.functional.normalize(k, dim=-1)

    #     attn = (q @ k.transpose(-2, -1)) * self.temperature # 4*8*8,4,32,32
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)

    #     x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    #     return x
        
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape  # 4*8*8,7*7,128
        # x=self.f2(x,mask)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv_band = self.qkv_band(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # 4*8*8,7*7,128 -> 4*8*8, 7*7, 3, 4, 128/4 -> 3, 4*8*8, 4, 7*7, 128/4
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) #256,4,49,49

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH  49,49,4
        
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww 4,49,49
        attn = attn + relative_position_bias.unsqueeze(0)#256,4,49,49

        if mask is not None:
            nW = mask.shape[0] # num_windows, Wh*Ww, Wh*Ww 8*8,49,49
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # 4,8*8,4,49,49 + 1,8*8,1,49,49 -> 4,8*8,4,49,49
            attn = attn.view(-1, self.num_heads, N, N) # 4*8*8,4,49,49
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x= (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # 4*8*8,4,49,49 @ 4*8*8, 4, 7*7, 32 ->  4*8*8,4,49,32 -> 4*8*8,49,4,32 -> 4*8*8,49,32*4
        # x=self.f2(x,mask)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape # 4*8*8,7*7,128
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # 4*8*8, 4, 7*7, 128/4
        q = q.transpose(-2, -1)
        # 4*8*8, 4, 32, 49
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature # 4*8*8,4,32,32
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # 4*8*8,4,32,49 -> 4*8*8,49,4,32 -> 4*8*8,49,128
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.se=SELayer(dim,1)# 尝试添加se模块
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        # self.Att=Attention_block(dim, dim, dim//4)

    def forward(self, x):
        H, W = self.input_resolution#56,56
        B, L, C = x.shape #4,3136,128
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)#4,56,56,128
        # x = self.se(x.permute(0,3,1,2)).permute(0,2,3,1)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C  4*8*8,7,7,128
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C  4*8*8,7*7,128

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C  4*8*8,7*7,128

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)#  4*8*8,7,7,128
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C 4,56,56,128

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C) #4,3136,128

        # # ATTENTION
        # x = self.Att(g=shortcut.view(B, H, W, C).permute(0,3,1,2),x=x.view(B, H, W, C).permute(0,3,1,2)).view(B, C, H * W).permute(0,2,1)
        # FFN
        x = shortcut + self.drop_path(x) #4,3136,128
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.view(B, L, C)
        return x
    
    
class WTPatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
    #J为分解的层次数,wave表示使用的变换方法
        self.xfm = DWTForward(J=1, mode='zero', wave='sym5')
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.formsize = nn.Conv2d(12,12,5,1,0)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        Yl, Yh = self.xfm(x)
        x = torch.cat([Yl,Yh[0][:,:,0,:,:],Yh[0][:,:,1,:,:],Yh[0][:,:,2,:,:]],1) # 1(B),12,104,104
        x = x.permute(0, 2, 3, 1)
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.formsize(x)
        x = self.norm(x)
        x = self.reduction(x) # B H/2*W/2 2*C

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class AttentionUnit(nn.Module):
    def __init__(self, channels):
        super(AttentionUnit, self).__init__()
        self.relu6 = nn.ReLU6()
        self.f = nn.Conv2d(channels, channels//2, (1, 1))
        self.g = nn.Conv2d(channels, channels//2, (1, 1))
        self.h = nn.Conv2d(channels, channels//2, (1, 1))

        self.out_conv = nn.Conv2d(channels//2, channels, (1, 1))
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, Fc, Fs):
        B, C, H, W = Fc.shape
        f_Fc = self.relu6(self.f(mean_variance_norm(Fc)))
        g_Fs = self.relu6(self.g(mean_variance_norm(Fs)))
        h_Fs = self.relu6(self.h(Fs))
        f_Fc = f_Fc.view(f_Fc.shape[0], f_Fc.shape[1], -1).permute(0, 2, 1)
        g_Fs = g_Fs.view(g_Fs.shape[0], g_Fs.shape[1], -1)

        Attention = self.softmax(torch.bmm(f_Fc, g_Fs))

        h_Fs = h_Fs.view(h_Fs.shape[0], h_Fs.shape[1], -1)

        Fcs = torch.bmm(h_Fs, Attention.permute(0, 2, 1))
        Fcs = Fcs.view(B, C//2, H, W)
        Fcs = self.relu6(self.out_conv(Fcs))

        return Fcs


class FuseUnit(nn.Module):
    def __init__(self, channels):
        super(FuseUnit, self).__init__()
        self.proj1 = nn.Conv2d(2*channels, channels, (1, 1))
        self.proj2 = nn.Conv2d(channels, channels, (1, 1))
        self.proj3 = nn.Conv2d(channels, channels, (1, 1))

        self.fuse1x = nn.Conv2d(channels, 1, (1, 1), stride = 1)
        self.fuse3x = nn.Conv2d(channels, 1, (3, 3), stride = 1)
        self.fuse5x = nn.Conv2d(channels, 1, (5, 5), stride = 1)

        self.pad3x = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pad5x = nn.ReflectionPad2d((2, 2, 2, 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, F1, F2):
        Fcat = self.proj1(torch.cat((F1, F2), dim=1))
        F1 = self.proj2(F1)
        F2 = self.proj3(F2)
        
        fusion1 = self.sigmoid(self.fuse1x(Fcat))      
        fusion3 = self.sigmoid(self.fuse3x(self.pad3x(Fcat)))
        fusion5 = self.sigmoid(self.fuse5x(self.pad5x(Fcat)))
        fusion = (fusion1 + fusion3 + fusion5) / 3

        return torch.clamp(fusion, min=0, max=1.0)*F1 + torch.clamp(1 - fusion, min=0, max=1.0)*F2 
     

class PAMA(nn.Module):
    def __init__(self, channels,resolution):
        super(PAMA, self).__init__()
        self.conv_in = nn.Conv2d(channels, channels, (3, 3), stride=1)
        self.attn = AttentionUnit(channels)
        self.fuse = FuseUnit(channels)
        self.conv_out = nn.Conv2d(channels, channels, (3, 3), stride=1)

        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.relu6 = nn.ReLU6()
        self.resolution = resolution
    
    def forward(self, Fc, Fs):
        H, W = self.resolution#56,56
        B1, L1, C1 = Fc.shape #128,3136,128
        assert L1 == H * W, "input feature has wrong size"
        Fc = Fc.view(B1, H, W, C1).permute(0,3,1,2)
        Fs = Fs.view(B1, H, W, C1).permute(0,3,1,2)

        Fc = self.relu6(self.conv_in(self.pad(Fc)))
        Fs = self.relu6(self.conv_in(self.pad(Fs)))
        Fcs = self.attn(Fc, Fs)
        Fcs = self.relu6(self.conv_out(self.pad(Fcs)))
        Fcs = self.fuse(Fc, Fcs)
        
        return Fcs
   

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        # self.pama=PAMA(dim//2,input_resolution)
        # patch merging layer
        if downsample is not None:
            if downsample is ConvDownsampler:
                self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            else:
                self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # pama模块
        # b,_,c=x.size()# 4 3136,128
        # x_h = self.pama(x[:,:,int(c/2):],x[:,:,:int(c/2)]).permute(0,2,3,1)
        # x=torch.cat([x[:,:,:int(c/2)],x_h.reshape(b,-1,int(c/2))],dim=2)
        #---------   
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # self.se=SELayer(in_chans,1)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj_2 = nn.Conv2d(in_chans*16, embed_dim, kernel_size=5,padding=2)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):# x:128,3,224,224
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.se(x)
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C [4,3136,128]
        
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class Cross_mix_o(nn.Module):
    def __init__(self, in_chans=3,image_size=7):
        super().__init__()
        self.proj1=nn.Conv2d(in_chans*2, in_chans, kernel_size=1, stride=1)
        self.proj2=nn.Conv2d(in_chans*2, 1, kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, upre,upost):# x:12,7,7,1024
        x = self.avg_pool(torch.cat([upre,upost],dim=3).permute((0,3,1,2))) #[12, 2048, 1, 1]   5:[12, 512, 1, 1]
        x = self.proj1(x)#12, 1024, 1, 1
        icha = self.sigmoid(x)#12, 1024, 1, 1
        if icha.dtype != upost.dtype:
            icha = icha.to(upost.dtype)
        uchapre=upost*icha.permute(0,2,3,1)+upre#[12, 7, 7, 1024]
        uchapost=upre*icha.permute(0,2,3,1)+upost#[12, 7, 7, 1024]
        x = self.proj2(torch.cat([uchapre,uchapost],dim=3).permute(0,3,1,2))
        ispa = self.sigmoid(x)#12, 1, 7, 7
        if ispa.dtype != upost.dtype:
            ispa = ispa.to(upost.dtype)
        uspapre=uchapost*ispa.permute(0,2,3,1)+upre
        uspapost=uchapre*ispa.permute(0,2,3,1)+upost
        return torch.cat([uspapre,uspapost],dim=3)#B,W,H, C [12, 7, 7, 2048]
  
class Cross_mix(nn.Module):
    def __init__(self, in_chans=3,image_size=7):
        super().__init__()
        self.chans=in_chans
        self.proj1=nn.Conv2d(in_chans*2, in_chans*2, kernel_size=1, stride=1)
        self.proj2=nn.Conv2d(in_chans*2, 1, kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, upre,upost):# x:12,7,7,1024
        x = self.avg_pool(torch.cat([upre,upost],dim=3).permute((0,3,1,2))) #[12, 2048, 1, 1]   5:[12, 512, 1, 1]
        x = self.proj1(x)#12, 1024, 1, 1
        icha = self.sigmoid(x)#12, 1024, 1, 1
        if icha.dtype != upost.dtype:
            icha = icha.to(upost.dtype)
        uchapre=upost*icha[:,:self.chans,:,:].permute(0,2,3,1)+upre#[12, 7, 7, 1024]
        uchapost=upre*icha[:,self.chans:,:,:].permute(0,2,3,1)+upost#[12, 7, 7, 1024]
        x = self.proj2(torch.cat([uchapre,uchapost],dim=3).permute(0,3,1,2))
        ispa = self.sigmoid(x)#12, 1, 7, 7
        if ispa.dtype != upost.dtype:
            ispa = ispa.to(upost.dtype)
        uspapre=uchapost*ispa.permute(0,2,3,1)+upre
        uspapost=uchapre*ispa.permute(0,2,3,1)+upost
        return torch.cat([uspapre,uspapost],dim=3)#B,W,H, C [12, 7, 7, 2048]

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return psi

    
class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False,old=6, **kwargs):
        super().__init__()
        #  双时相的尝试
        # in_chans=6

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        if old in [1,4,5,6,7,8,9]:
            self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        elif old==2:
            self.num_features = int(256 * 2 ** (self.num_layers - 1))
        else:
            self.num_features = int((56 * 64) * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.old = old
        # split image into non-overlapping patches
        if old in [1,4,5,6,7,8,9]:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
        elif old==2:
            self.patch_embed = PatchEmbed(
                img_size=56, patch_size=1, in_chans=2688, embed_dim=256,
                norm_layer=norm_layer if self.patch_norm else None)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=64,
                norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            if old in [1 ,4,5,6,9]:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            elif old==2:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, 256))
            else:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 56*56, 56 * 64))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        if old in [1,4 ,5,6,7,8,9]:
            self.layers = nn.ModuleList()
            self.mix_layers = nn.ModuleList()
            self.compose=nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                    patches_resolution[1] // (2 ** i_layer)),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
                if len(self.mix_layers)==3:
                    mix_layer = Cross_mix(int(embed_dim * 2 ** (i_layer-1)),patches_resolution[0] // (2 ** i_layer))
                    self.compose.append(nn.Conv2d(int(embed_dim * 2 ** (i_layer-1))*4,int(embed_dim * 2 ** (i_layer-1))*2,3,1,1))
                else:
                    mix_layer = Cross_mix(int(embed_dim * 2 ** (i_layer)),patches_resolution[0] // (2 ** i_layer))
                    self.compose.append(nn.Conv2d(int(embed_dim * 2 ** (i_layer))*4,int(embed_dim * 2 ** (i_layer))*2,3,1,1))

                self.layers.append(layer)
                self.mix_layers.append(mix_layer)
        elif old==2:
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(256 * 2 ** i_layer),
                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                    patches_resolution[1] // (2 ** i_layer)),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,# 是否要改存疑
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
                self.layers.append(layer)
        elif old==3:
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int((56 * 64) * 2 ** i_layer),
                                input_resolution=(56 // (2 ** i_layer),
                                                    56 // (2 ** i_layer)),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,# 是否要改存疑
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
                self.layers.append(layer)
            # 波段太多了
            # self.proj = nn.Conv2d(56 * 64, 56 * 32, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(self.num_features)
        if old in [6,9]:
            self.norm=nn.ModuleList()
            for i in range(self.num_layers):
                if i==(self.num_layers-1):
                    self.norm.append(norm_layer(int(embed_dim * 2 ** (i))))
                else:
                    self.norm.append(norm_layer(int(embed_dim * 2 ** (i+1))))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if old in [4,5,6,9]:
            temp_num_features=self.num_features*2
        else:
            temp_num_features=self.num_features
        self.head = nn.Linear(temp_num_features, num_classes) if num_classes > 0 else nn.Identity()
        if old in [6,9]:
            self.head=nn.ModuleList()
            # self.head_a=nn.ModuleList()
            for i in range(self.num_layers):
                if i==(self.num_layers-1):
                    self.head.append(nn.Linear(4*int(embed_dim * 2 ** (i-2)), num_classes))
                    # self.head_a.append(nn.Linear(4*int(embed_dim * 2 ** (i-2)), num_classes))
                else:
                    self.head.append(nn.Linear(4*int(embed_dim * 2 ** (i-1)), num_classes))
                    # self.head_a.append(nn.Linear(4*int(embed_dim * 2 ** (i-1)), num_classes))
        
        self.se=SELayer(49*1)
        # self.pama = PAMA(embed_dim//2,(patches_resolution[0],patches_resolution[1]))
        self.apply(self._init_weights)
        self.crossmix=Cross_mix(1024,7)
        if old==8:
            self.crossmix=Cross_mix_o(1024,7)
        self.crossmix_front=Cross_mix(3,224)
        self.conv_downsampling = nn.Conv2d(3,3,kernel_size=2,stride=2)
        pos_dim=8
        # self.pos_conv=nn.Conv1d(2,pos_dim,1)
        # self.pos_mlp=Mlp(pos_dim*8, 64,4,drop=0.1)
        # self.pos_norm = nn.BatchNorm1d(pos_dim)
        # self.pos_act=nn.ReLU()
        # self.pos_drop2 = nn.Dropout(p=0.1)
        self.att01=Attention_block(6,1,4)
        # self.att01=Attention_block(6,2,4)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features_old1(self, x):# x:[4,3,224,224]
        x = self.patch_embed(x)    # x:[4,3136,128]
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x) # x:[4,3136,128]
        '''
        # pama模块
        b,_,c=x.size()# 4 3136,128
        x_h = self.pama(x[:,:,int(c/2):],x[:,:,:int(c/2)]).permute(0,2,3,1)
        x=torch.cat([x[:,:,:int(c/2)],x_h.reshape(b,-1,int(c/2))],dim=2)
        ''' 
        i=0
        for layer in self.layers:
            x = layer(x) #torch.Size([16, 784, 256]) 3,4th 16,49,1024
            # B,L_temp,C=x.shape
            # L_temp= int(np.sqrt(L_temp)) # 16,28,28,128
            # x = self.mix_layers[i](x[:,:,:C//2].reshape((B,L_temp,L_temp,C//2)),x[:,:,(C//2):].reshape((B,L_temp,L_temp,C//2))).reshape((B,-1,C))#12, 49, 2048
            i+=1

        x = self.norm(x)  # B L C 形状不变  12,49,1024
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1) #
        return x

    def forward_features_old2(self, x):# x:[4,3,224,224]
        B,C,_,_ = x.shape
        x1 = x[:,:3].reshape((B,C,56,-1)).unsqueeze(3)# b,3,196,1,256
        x2 = x[:,3:].reshape((B,C,56,-1)).unsqueeze(2)# b,3,1,196,256
        x = x1-x2  # b,3,196,196,256
        _,_,H,W,dim1 = x.shape
        x = x.permute(0,1,4,2,3).reshape((B,C*dim1,H,W))
        x = self.patch_embed(x)    # x:[4,1024,588]
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x) # x:[4,3136,128]
        # pama模块
        # b,_,c=x.size()# 4 3136,128
        # x_h = self.pama(x[:,:,int(c/2):],x[:,:,:int(c/2)]).permute(0,2,3,1)
        # x=torch.cat([x[:,:,:int(c/2)],x_h.reshape(b,-1,int(c/2))],dim=2)
        #---------
        for layer in self.layers:
            x = layer(x) #

        x = self.norm(x)  # B L C 形状不变
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1) #
        return x

    def forward_features_old3(self, x):# x:[4,3,224,224]
        B,C,_,_ = x.shape
        x1 = x[:,:3]
        x2 = x[:,3:]
        x1 = self.patch_embed(x1).reshape((B,7,8,8,7,-1)).permute(0,1,3,2,4,5).view(B,56,8,7,-1).unsqueeze(2)#[4,3136,96]
        x2 = self.patch_embed(x2).reshape((B,7,8,8,7,-1)).permute(0,1,3,2,4,5).view(B,56,8,7,-1).unsqueeze(1)#[4,3136,96]
        x = x1 - x2 # B, 56, 56, 8, 7, 96
        x = x.reshape(B, 56 * 56, 56 * 64)
        
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x) # x:[4,3136,128]
        # pama模块
        # b,_,c=x.size()# 4 3136,128
        # x_h = self.pama(x[:,:,int(c/2):],x[:,:,:int(c/2)]).permute(0,2,3,1)
        # x=torch.cat([x[:,:,:int(c/2)],x_h.reshape(b,-1,int(c/2))],dim=2)
        #---------
        for layer in self.layers:
            x = layer(x) #

        x = self.norm(x)  # B L C 形状不变
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1) #
        return x
    
    def forward_features_old4(self, x):# x:[4,3,224,224]
        x = self.patch_embed(x)    # x:[4,3136,128]
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x) # x:[4,3136,128]

        for layer in self.layers:
            x = layer(x) #

        x = self.norm(x)  # B L C 形状不变
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1) #
        return x

    def forward_features_old5(self, x):# x:[4,3,224,224]
        x1 = self.patch_embed(x[:,:3,:,:])    # x:[4,3136,128]
        x2 = self.patch_embed(x[:,3:,:,:])

        if self.ape:
            x1 = x1 + self.absolute_pos_embed
            x2 = x2 + self.absolute_pos_embed
        x1 = self.pos_drop(x1) # x:[4,3136,128]
        x2 = self.pos_drop(x2) # x:[4,3136,128]
        i=0
        for layer in self.layers:
            x1 = layer(x1) #[12, 784, 256]   [12, 196, 512]   [12, 49, 1024] [12, 49, 1024]
            x2 = layer(x2) #[12, 784, 256]   [12, 196, 512]   [12, 49, 1024] [12, 49, 1024]
            B,L_temp,C=x1.shape
            L_temp= int(np.sqrt(L_temp))
            x3 = self.mix_layers[i](x1.reshape((B,L_temp,L_temp,C)),x2.reshape((B,L_temp,L_temp,C))).reshape((B,-1,C*2))#12, 49, 2048
            _,_,C1=x3.shape #[12, 784, 512] [12, 196, 1024]
            x1=x3[:,:,:C1//2]
            x2=x3[:,:,C1//2:]
            i+=1
        
        x1 = self.norm(x1)  # B L C 形状不变
        x2 = self.norm(x2)  # B L C 形状不变
        x = torch.cat([x1,x2],dim=2)#[12, 49, 2048]
        x = self.avgpool(x.transpose(1, 2))  # B C 1 [12, 2048, 1]
        x = torch.flatten(x, 1) #[12, 2048] 
        x = self.head(x)
        return x
    
    def forward_features_old6(self, x):# x:[4,3,224,224]
        x = self.patch_embed(x)    # x:[4,3136,128] 14*14*4*4 56*56
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x) # x:[4,3136,128]
        feat_list=[]
        i=0 
        for layer in self.layers:
            x = layer(x) #torch.Size([16, 784, 256]) 3,4th 16,49,1024
            B,L_temp,C=x.shape
            L_temp= int(np.sqrt(L_temp)) # 16,28,28,128
            x = self.mix_layers[i](x[:,:,:C//2].reshape((B,L_temp,L_temp,C//2)),x[:,:,(C//2):].reshape((B,L_temp,L_temp,C//2))).reshape((B,-1,C))#12, 49, 2048
            i+=1
            feat_list.append(x)
            
        res_list=[]
        # res_list_a=[]
        for_tsne=[]
        for i in range(len(feat_list)):
            x=feat_list[i]
            norm=self.norm[i]
            head=self.head[i]
            # heada=self.head_a[i]
            x = norm(x)  # B L C 形状不变
            x = self.avgpool(x.transpose(1, 2))  # B C 1 [12, 2048, 1]
            x = torch.flatten(x, 1) #[12, 512] 
            for_tsne.append(x)
            res_list.append(head(x))# x:16 256
            # res_list_a.append(heada(x))# x:16 256

            
        # res_list=torch.stack(res_list,dim=0)
        # x,_ = torch.max(res_list,dim=0) 
        x=sum(res_list)/ len(res_list)
        # x_a=sum(res_list_a)/ len(res_list_a)
        # x=(x+x_a).softmax(dim=1)
        # x=(x+x_a)/2
        return x
    
    def forward_features_old6_2(self, x):# x:[4,3,224,224]
        x = self.patch_embed(x)    # x:[4,3136,128]
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x) # x:[4,3136,128]
        feat_list=[]
        i=0
        for layer in self.layers:
            x = layer(x) #torch.Size([16, 784, 256]) 3,4th 16,49,1024
            B,L_temp,C=x.shape
            L_temp= int(np.sqrt(L_temp)) # 16,28,28,128
            x = self.mix_layers[i](x[:,:,:C//2].reshape((B,L_temp,L_temp,C//2)),x[:,:,(C//2):].reshape((B,L_temp,L_temp,C//2))).reshape((B,-1,C))#12, 49, 2048
            i+=1
            feat_list.append(x)
            
        return feat_list
    
    def forward_head_old6(self,feat_list1,feat_list2):
        feat_list=[]
        for i in range(len(feat_list1)):
            f1=feat_list1[i]
            f2=feat_list2[i]
            com=self.compose[i]
            B,S,C=f1.shape
            w=int(np.sqrt(S))
            feat_list.append(com(torch.cat((f1,f2),2).view(B,w,w,2*C).permute(0,3,1,2)).permute(0,2,3,1).view(B,-1,C))
        res_list=[]
        # res_list_a=[]
        for_tsne=[]
        for i in range(len(feat_list)):
            x=feat_list[i]
            norm=self.norm[i]
            head=self.head[i]
            # heada=self.head_a[i]
            x = norm(x)  # B L C 形状不变
            x = self.avgpool(x.transpose(1, 2))  # B C 1 [12, 2048, 1]
            x = torch.flatten(x, 1) #[12, 512] 
            for_tsne.append(x)
            res_list.append(head(x))# x:16 256
            # res_list_a.append(heada(x))# x:16 256

            
        # res_list=torch.stack(res_list,dim=0)
        # x,_ = torch.max(res_list,dim=0) 
        x=sum(res_list)/ len(res_list)
        # x_a=sum(res_list_a)/ len(res_list_a)
        # x=(x+x_a).softmax(dim=1)
        # x=(x+x_a)/2
        return x
        
    def forward_features_old6_3(self, x):# x:[4,3,224,224]
        x = self.patch_embed(x)    # x:[4,3136,128]
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x) # x:[4,3136,128]
        feat_list=[]
        i=0
        for layer in self.layers:
            x = layer(x) #torch.Size([16, 784, 256]) 3,4th 16,49,1024
            i+=1
            feat_list.append(x)
        res_list=[]
        for i in range(len(feat_list)):
            x=feat_list[i]
            norm=self.norm[i]
            head=self.head[i]
            x = norm(x)  # B L C 形状不变
            x = self.avgpool(x.transpose(1, 2))  # B C 1 [12, 2048, 1]
            x = torch.flatten(x, 1) #[12, 512] 
            res_list.append(head(x))# x:16 256
            
        # res_list=torch.stack(res_list,dim=0)
        # x,_ = torch.max(res_list,dim=0) 
        x=sum(res_list)/ len(res_list)
        return x

    def forward_features_old7(self, x):# x:[4,3,224,224]
        x = self.patch_embed(x)    # x:[4,3136,128]
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x) # x:[4,3136,128]
        feat_list=[]
        i=0
        for layer in self.layers:
            x = layer(x) #torch.Size([16, 784, 256]) 3,4th 16,49,1024
            B,L_temp,C=x.shape
            L_temp= int(np.sqrt(L_temp)) # 16,28,28,128
            x = self.mix_layers[i](x[:,:,:C//2].reshape((B,L_temp,L_temp,C//2)),x[:,:,(C//2):].reshape((B,L_temp,L_temp,C//2))).reshape((B,-1,C))#12, 49, 2048
            i+=1
            feat_list.append(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1) 
        x = self.head(x)
        return x
    
    def forward_features_old8(self, x):# x:[4,3,224,224]
        x = self.patch_embed(x)    # x:[4,3136,128]
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x) # x:[4,3136,128]
        feat_list=[]
        i=0
        for layer in self.layers:
            x = layer(x) #torch.Size([16, 784, 256]) 3,4th 16,49,1024
            B,L_temp,C=x.shape
            L_temp= int(np.sqrt(L_temp)) # 16,28,28,128
            x = self.mix_layers[i](x[:,:,:C//2].reshape((B,L_temp,L_temp,C//2)),x[:,:,(C//2):].reshape((B,L_temp,L_temp,C//2))).reshape((B,-1,C))#12, 49, 2048
            i+=1
            feat_list.append(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1,) 
        x = self.head(x)
        return x
    
    def forward_features_old9(self, x, pos):# x:[4,3,224,224] add pos info
        res1=self.pos_mlp(self.pos_drop2(torch.flatten(self.pos_act(self.pos_norm(self.pos_conv(pos.float()))),1)))
        x = self.patch_embed(x)    # x:[4,3136,128]
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x) # x:[4,3136,128]
        feat_list=[]
        i=0
        for layer in self.layers:
            x = layer(x) #torch.Size([16, 784, 256]) 3,4th 16,49,1024
            B,L_temp,C=x.shape
            L_temp= int(np.sqrt(L_temp)) # 16,28,28,128
            x = self.mix_layers[i](x[:,:,:C//2].reshape((B,L_temp,L_temp,C//2)),x[:,:,(C//2):].reshape((B,L_temp,L_temp,C//2))).reshape((B,-1,C))#12, 49, 2048
            i+=1
            feat_list.append(x)
            
        res_list=[]
        # res_list_a=[]
        for_tsne=[]
        for i in range(len(feat_list)):
            x=feat_list[i]
            norm=self.norm[i]
            head=self.head[i]
            # heada=self.head_a[i]
            x = norm(x)  # B L C 形状不变
            x = self.avgpool(x.transpose(1, 2))  # B C 1 [12, 2048, 1]
            x = torch.flatten(x, 1) #[12, 512] 
            for_tsne.append(x)
            res_list.append(head(x))# x:16 256
        x=sum(res_list)/ len(res_list)
        x=sum([x,res1])/2
        return x
    
    def forward(self, x, pos=None):
        if self.old==1:
            B,C,W,H=x.shape
            # x = self.crossmix_front(x[:,:3,:,:].permute((0,2,3,1)),x[:,3:,:,:].permute((0,2,3,1))).permute((0,3,1,2))
            x = self.forward_features_old1(x[:,:6])
            x = self.head(x)
     
        elif self.old==2:
            x = self.forward_features_old2(x)
            x = self.head(x)
        elif self.old==3:
            x = self.forward_features_old3(x)
            x = self.head(x)
        elif self.old==4:
            x1=self.forward_features_old4(x[:,:3,:,:])
            x2=self.forward_features_old4(x[:,3:,:,:])
            B,_,C=x1.shape
            x3 = self.crossmix(x1.reshape((B,7,7,C)),x2.reshape((B,7,7,C))).reshape((B,-1,C*2))#12, 49, 2048
            x = self.avgpool(x3.transpose(1, 2))  # B C 1 [12, 2048, 1]
            x = torch.flatten(x, 1) #[12, 2048]
            x = self.head(x)
        elif self.old==5:
            x = self.forward_features_old5(x)
        elif self.old==6:
            # _,C,_,_ = x.shape
            # f1 = self.forward_features_old6(x[:,:6])
            # if C==8:
            #     env_factor=x[:,6:8]# all
            #     # env_factor=x[:,6:7] #dem
            #     # env_factor=x[:,7:8]#lc
            #     psi=self.att01(env_factor,x[:,:6])
            #     x=x[:,:6]*psi
            # f2 = self.forward_features_old6(x)
            # x=(f1+f2)/2

            x = self.forward_features_old6_3(x[:,:6])
            # x = self.forward_features_old6(x[:,:6])
        elif self.old==7:
            x = self.forward_features_old7(x[:,:6])
        elif self.old==8:
            x = self.forward_features_old8(x[:,:6])
        elif self.old==9:
            x = self.forward_features_old9(x,pos)
        return x
        

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

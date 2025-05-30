# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from abc import abstractmethod
from .basic_ops import normalization
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
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
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
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
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple), B_ x H x N x C

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0).to(attn.dtype)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        patch_norm (bool, optional): True, GroupNorm32
        in_chans (int): unused. Number of input image channels. Default: 3.
    """
    def __init__(
            self,
            in_chans,
            img_size=224,
            patch_size=4,
            embed_dim=96,
            patch_norm=False,
            ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if patch_norm:
            self.norm = normalization(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: B x C x H x W
        output: B x embed_dim x Ph x Pw, Ph = H // patch_size

        """
        x = self.proj(x)  # B x embed_dim x Ph x Pw
        x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r""" Patch to Image.

    Args:
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, out_chans, embed_dim=96, patch_norm=False):
        super().__init__()
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(embed_dim, out_chans, kernel_size=1, stride=1)
        if patch_norm:
            self.norm = normalization(out_chans)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        '''
        Args:
            x: B x C x Ph x Pw
        out: B x C x Ph x Pw
        '''
        x = self.norm(self.proj(x))
        return x
 

class SwinTransformerBlock_AdaFM(nn.Module):
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
                 act_layer=nn.GELU, norm_layer=normalization, emb_channels=160*4, fft_patch_size=8, **kwargs):
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

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.fft_patch_size = min(fft_patch_size, min(input_resolution))
        self.adaLN_scale_msa = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.fft_patch_size * (self.fft_patch_size // 2 + 1))
        )
        self.adaLN_scale_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.fft_patch_size * (self.fft_patch_size // 2 + 1))
        )

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, 1, H, W))  # 1 H W 1
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

        mask_windows = window_partition(img_mask, self.window_size).permute(0,2,3,1).contiguous()  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # nW, window_size*window_size

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, t):
        '''
        Args:
            x: B x C x Ph x Pw, Ph = H // patch_size
        Out:
            x: B x (H*W) x C
        '''
        B, C, Ph, Pw = x.shape
        x_size = (Ph, Pw)
        x_type = x.dtype

        scale_msa = self.adaLN_scale_msa(t).reshape(t.shape[0], 1, 1, 1, self.fft_patch_size, self.fft_patch_size // 2 + 1)
        scale_mlp = self.adaLN_scale_mlp(t).reshape(t.shape[0], 1, 1, 1, self.fft_patch_size, self.fft_patch_size // 2 + 1)

        shortcut = x
        x = self.norm1(x)

        # AdaFM
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.fft_patch_size,
                            patch2=self.fft_patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft *= scale_msa
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.fft_patch_size, self.fft_patch_size)).to(x_type)
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.fft_patch_size,
                      patch2=self.fft_patch_size)

        # cyclic shift, shifted_x: B x C x Ph x Pw
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        # partition windows, NW: number of windows, Ws: window size
        x_windows = window_partition(shifted_x, self.window_size)  # (NW*B) x Ws x Ws x C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (NW*B) x (Ws*Ws) x C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask.to(x.dtype))  # (NW*B) x (Ws*Ws) x C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device, x.dtype))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Ph, Pw)  # B x C x Ph x Pw

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)
        
        shortcut = x
        x = self.norm2(x)
        
        # AdaFM
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.fft_patch_size,
                            patch2=self.fft_patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft *= scale_mlp
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.fft_patch_size, self.fft_patch_size)).to(x_type)
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.fft_patch_size,
                      patch2=self.fft_patch_size)
        
        # FFN
        x = shortcut + self.drop_path(self.mlp(x))
        return x


class BasicLayer(TimestepBlock):
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
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size (int): image resolution. Defaulr: 224
        patch_size (int): patch resolution. Default: 1
        patch_norm (bool): patch normalization. Default: False
    """
    def __init__(
            self,
            in_chans,
            embed_dim,
            num_heads,
            window_size,
            depth=2,
            img_size=224,
            patch_size=4,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=normalization,
            use_checkpoint=False,
            patch_norm=True,
            time_embed_dim=160*4,
            patch_emb=True,
            **kwargs,
             ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.patch_emb = patch_emb
        self.patch_embed = PatchEmbed(
                in_chans=in_chans,
                embed_dim=embed_dim,
                img_size=img_size,
                patch_size=patch_size,
                patch_norm=patch_norm,
                )
        self.patch_unembed = PatchUnEmbed(
            out_chans=in_chans,
            embed_dim=embed_dim,
            patch_norm=patch_norm,
            )
        if patch_emb is False:
            del self.patch_embed
            del self.patch_unembed

        input_resolution = self.patch_embed.patches_resolution
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList([
            SwinTransformerBlock_AdaFM(
                        dim=embed_dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        emb_channels=time_embed_dim,
                        **kwargs,
                                 )
            for i in range(depth)])

    def forward(self, x, t):
        '''
        Args:
            x: B x C x H x W, H,W: height and width after patch embedding
            x_size: (H, W)
        Out:
            x: B x H x W x C
        '''
        x = self.patch_embed(x) if self.patch_emb else x # B x embed_dim x Ph x Pw
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, t)
            else:
                x = blk(x, t)
        x = self.patch_unembed(x) if self.patch_emb else x # B x C x Ph x Pw
        return x

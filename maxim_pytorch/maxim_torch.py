# -*- coding: utf-8 -*-
import einops
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


def nearest_downsample(x, ratio):
    n, c, h, w = x.shape
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    h_index = np.floor((np.arange(new_h) + 0.5) / ratio)
    w_index = np.floor((np.arange(new_w) + 0.5) / ratio)
    out = x[:, :, h_index, :]
    out = out[:, :, :, w_index]
    return out


class Layer_norm_process(nn.Module):  # n, h, w, c
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(c), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(c), requires_grad=True)
        self.eps = eps

    def forward(self, feature):
        var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
        mean = var_mean[1]
        var = var_mean[0]
        # layer norm process
        feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + self.eps)
        gamma = self.gamma.expand_as(feature)
        beta = self.beta.expand_as(feature)
        feature = feature * gamma + beta
        return feature


def block_images_einops(x, patch_size):  # n, h, w, c
    """Image to patches."""
    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
        x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x


class UpSampleRatio_4(nn.Module):  # input shape: n,c,h,w.    c-->4c
    """Upsample features given a ratio > 0."""

    def __init__(self, features, b=0, ratio=1., use_bias=True):
        super().__init__()
        self.features = features
        self.ratio = ratio
        self.bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, 4 * self.features, kernel_size=(1, 1), stride=1, bias=self.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear', antialias=True)
        x = self.Conv_0(x)
        return x


class UpSampleRatio_2(nn.Module):  # input shape: n,c,h,w.    c-->2c
    """Upsample features given a ratio > 0."""

    def __init__(self, features, b=0, ratio=1., use_bias=True):
        super().__init__()
        self.features = features
        self.ratio = ratio
        self.bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, 2 * self.features, kernel_size=(1, 1), stride=1, bias=self.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear', antialias=True)
        x = self.Conv_0(x)
        return x


class UpSampleRatio(nn.Module):  # input shape: n,c,h,w.    c-->c
    """Upsample features given a ratio > 0."""

    def __init__(self, features, b=0, ratio=1., use_bias=True):
        super().__init__()
        self.features = features
        self.ratio = ratio
        self.bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, self.features, kernel_size=(1, 1), stride=1, bias=self.bias)

    def forward(self, x):
        x = self.Conv_0(x)
        return x


class UpSampleRatio_1_2(nn.Module):  # input shape: n,c,h,w.    c-->c/2
    """Upsample features given a ratio > 0."""

    def __init__(self, features, b=0, ratio=1., use_bias=True):
        super().__init__()
        self.features = features
        self.ratio = ratio
        self.bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, self.features // 2, kernel_size=(1, 1), stride=1, bias=self.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear', antialias=True)
        x = self.Conv_0(x)
        return x


class UpSampleRatio_1_4(nn.Module):  # input shape: n,c,h,w.    c-->c/4
    """Upsample features given a ratio > 0."""

    def __init__(self, features, b=0, ratio=1., use_bias=True):
        super().__init__()
        self.features = features
        self.ratio = ratio
        self.bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, self.features // 4, kernel_size=(1, 1), stride=1, bias=self.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear', antialias=True)
        x = self.Conv_0(x)
        return x


class BlockGatingUnit(nn.Module):  # input shape: n (gh gw) (fh fw) c
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second last.
    If applied on other dims, you should swapaxes first.
    """

    def __init__(self, c, n, use_bias=True):
        super().__init__()
        self.c = c
        self.n = n
        self.use_bias = use_bias
        self.Dense_0 = nn.Linear(self.n, self.n, self.use_bias)
        self.intermediate_layernorm = Layer_norm_process(self.c // 2)

    def forward(self, x):
        c = x.size(-1)
        c = c // 2  # split size
        u, v = torch.split(x, c, dim=-1)
        v = self.intermediate_layernorm(v)
        v = v.permute(0, 1, 3, 2)  # n, (gh gw), c/2, (fh fw)
        v = self.Dense_0(v)  # apply fc on the last dimension (fh fw)
        v = v.permute(0, 1, 3, 2)  # n (gh gw) (fh fw) c/2
        return u * (v + 1.)


class GridGatingUnit(nn.Module):  # input shape: n (gh gw) (fh fw) c
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second.
    If applied on other dims, you should swapaxes first.
    """

    def __init__(self, c, n, use_bias=True):
        super().__init__()
        self.c = c
        self.n = n
        self.use_bias = use_bias
        self.intermediate_layernorm = Layer_norm_process(self.c // 2)
        self.Dense_0 = nn.Linear(self.n, self.n, self.use_bias)

    def forward(self, x):
        c = x.size(-1)
        c = c // 2  # split size
        u, v = torch.split(x, c, dim=-1)
        v = self.intermediate_layernorm(v)
        v = v.permute(0, 3, 2, 1)  # n, c/2, (fh fw) (gh gw)
        v = self.Dense_0(v)  # apply fc on the last dimension (gh gw)
        v = v.permute(0, 3, 2, 1)  # n (gh gw) (fh fw) c/2
        return u * (v + 1.)


class GridGmlpLayer(nn.Module):  # input shape: n, h, w, c
    """Grid gMLP layer that performs global mixing of tokens."""

    def __init__(self, grid_size, num_channels, use_bias=True, factor=2, dropout_rate=0):
        super().__init__()
        self.grid_size = grid_size
        self.gh = grid_size[0]
        self.gw = grid_size[1]
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.factor = factor
        self.dropout_rate = dropout_rate
        self.LayerNorm = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.factor, self.use_bias)  # c->c*factor
        self.gelu = nn.GELU(approximate='tanh')
        self.GridGatingUnit = GridGatingUnit(self.num_channels * self.factor,
                                             n=self.gh * self.gw)  # number of channels????????????????
        self.out_project = nn.Linear(self.num_channels * self.factor // 2, self.num_channels,
                                     self.use_bias)  # c*factor->c
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        n, h, w, num_channels = x.shape
        fh, fw = h // self.gh, w // self.gw
        x = block_images_einops(x, patch_size=(fh, fw))  # n (gh gw) (fh fw) c
        # gMLP1: Global (grid) mixing part, provides global grid communication.
        y = self.LayerNorm(x)
        y = self.in_project(y)  # channel proj
        y = self.gelu(y)
        y = self.GridGatingUnit(y)
        y = self.out_project(y)
        y = self.dropout(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(self.gh, self.gw), patch_size=(fh, fw))
        return x


class BlockGmlpLayer(nn.Module):  # input shape: n, h, w, c
    """Block gMLP layer that performs local mixing of tokens."""

    def __init__(self, block_size, num_channels, use_bias=True, factor=2, dropout_rate=0):
        super().__init__()
        self.block_size = block_size
        self.fh = self.block_size[0]
        self.fw = self.block_size[1]
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.factor = factor
        self.dropout_rate = dropout_rate
        self.LayerNorm = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.factor, self.use_bias)  # c->c*factor
        self.gelu = nn.GELU(approximate='tanh')
        self.BlockGatingUnit = BlockGatingUnit(self.num_channels * self.factor,
                                               n=self.fh * self.fw)  # number of channels????????????????
        self.out_project = nn.Linear(self.num_channels * self.factor // 2, self.num_channels,
                                     self.use_bias)  # c*factor->c
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        _, h, w, _ = x.shape
        gh, gw = h // self.fh, w // self.fw
        x = block_images_einops(x, patch_size=(self.fh, self.fw))  # n (gh gw) (fh fw) c
        # gMLP2: Local (block) mixing part, provides local block communication.
        y = self.LayerNorm(x)
        y = self.in_project(y)  # channel proj
        y = self.gelu(y)
        y = self.BlockGatingUnit(y)
        y = self.out_project(y)
        y = self.dropout(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(self.fh, self.fw))
        return x


class MlpBlock(nn.Module):  # input shape: n, h, w, c
    """A 1-hidden-layer MLP block, applied over the last dimension."""

    def __init__(self, mlp_dim, dropout_rate=0., use_bias=True):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.Dense_0 = nn.Linear(self.mlp_dim, self.mlp_dim, bias=self.use_bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.dropout = nn.Dropout(self.dropout_rate)
        self.Dense_1 = nn.Linear(self.mlp_dim, self.mlp_dim, bias=self.use_bias)

    def forward(self, x):
        x = self.Dense_0(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.Dense_1(x)
        return x


class CALayer(nn.Module):  # input shape: n, h, w, c
    """Squeeze-and-excitation block for channel attention.
    ref: https://arxiv.org/abs/1709.01507
    """

    def __init__(self, features, reduction=4, use_bias=True):
        super().__init__()
        self.features = features
        self.reduction = reduction
        self.use_bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, self.features // self.reduction, kernel_size=(1, 1), stride=1,
                                bias=self.use_bias)  # 1*1 conv
        self.relu = nn.ReLU()
        self.Conv_1 = nn.Conv2d(self.features // self.reduction, self.features, kernel_size=(1, 1), stride=1,
                                bias=self.use_bias)  # 1*1 conv
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)  # n, c, h, w
        y = torch.mean(y, dim=(2, 3), keepdim=True)  # keep dimensions for element product in the last step
        y = self.Conv_0(y)
        y = self.relu(y)
        y = self.Conv_1(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 3, 1)  # n, h, w, c
        return x * y


class GetSpatialGatingWeights(nn.Module):  # n, h, w, c
    """Get gating weights for cross-gating MLP block."""

    def __init__(self, num_channels, grid_size, block_size, input_proj_factor=2, use_bias=True, dropout_rate=0):
        super().__init__()
        self.num_channels = num_channels
        self.grid_size = grid_size
        self.block_size = block_size
        self.gh = self.grid_size[0]
        self.gw = self.grid_size[1]
        self.fh = self.block_size[0]
        self.fw = self.block_size[1]
        self.input_proj_factor = input_proj_factor
        self.use_bias = use_bias
        self.drop = dropout_rate
        self.LayerNorm_in = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.input_proj_factor, bias=self.use_bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.Dense_0 = nn.Linear(self.gh * self.gw, self.gh * self.gw, bias=self.use_bias)
        self.Dense_1 = nn.Linear(self.fh * self.fw, self.fh * self.fw, bias=self.use_bias)
        self.out_project = nn.Linear(self.num_channels * self.input_proj_factor, self.num_channels, bias=self.use_bias)
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        _, h, w, _ = x.shape
        # input projection
        x = self.LayerNorm_in(x)
        x = self.in_project(x)  # channel projection
        x = self.gelu(x)
        c = x.size(-1) // 2
        u, v = torch.split(x, c, dim=-1)
        # get grid MLP weights
        fh, fw = h // self.gh, w // self.gw
        u = block_images_einops(u, patch_size=(fh, fw))  # n, (gh gw) (fh fw) c
        u = u.permute(0, 3, 2, 1)  # n, c, (fh fw) (gh gw)
        u = self.Dense_0(u)
        u = u.permute(0, 3, 2, 1)  # n, (gh gw) (fh fw) c
        u = unblock_images_einops(u, grid_size=(self.gh, self.gw), patch_size=(fh, fw))
        # get block MLP weights
        gh, gw = h // self.fh, w // self.fw
        v = block_images_einops(v, patch_size=(self.fh, self.fw))  # n, (gh gw) (fh fw) c
        v = v.permute(0, 1, 3, 2)  # n (gh gw) c (fh fw)
        v = self.Dense_1(v)
        v = v.permute(0, 1, 3, 2)  # n, (gh gw) (fh fw) c
        v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(self.fh, self.fw))

        x = torch.cat([u, v], dim=-1)
        x = self.out_project(x)
        x = self.dropout(x)
        return x


class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):  # input shape: n, h, w, c
    """The multi-axis gated MLP block."""

    def __init__(self, block_size, grid_size, num_channels, input_proj_factor=2, block_gmlp_factor=2,
                 grid_gmlp_factor=2, use_bias=True, dropout_rate=0.):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.input_proj_factor = input_proj_factor
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.use_bias = use_bias
        self.drop = dropout_rate
        self.LayerNorm_in = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.input_proj_factor, bias=self.use_bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.GridGmlpLayer = GridGmlpLayer(grid_size=self.grid_size,
                                           num_channels=self.num_channels * self.input_proj_factor // 2,
                                           use_bias=self.use_bias, factor=self.grid_gmlp_factor)
        self.BlockGmlpLayer = BlockGmlpLayer(block_size=self.block_size,
                                             num_channels=self.num_channels * self.input_proj_factor // 2,
                                             use_bias=self.use_bias, factor=self.block_gmlp_factor)
        self.out_project = nn.Linear(self.num_channels * self.input_proj_factor, self.num_channels, bias=self.use_bias)
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        shortcut = x
        x = self.LayerNorm_in(x)
        x = self.in_project(x)
        x = self.gelu(x)
        c = x.size(-1) // 2
        u, v = torch.split(x, c, dim=-1)
        # grid gMLP
        u = self.GridGmlpLayer(u)
        # block gMLP
        v = self.BlockGmlpLayer(v)
        # out projection
        x = torch.cat([u, v], dim=-1)
        x = self.out_project(x)
        x = self.dropout(x)
        x = x + shortcut
        return x


class RCAB(nn.Module):  # input shape: n, h, w, c
    """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""

    def __init__(self, features, reduction=4, lrelu_slope=0.2, use_bias=True):
        super().__init__()
        self.features = features
        self.reduction = reduction
        self.lrelu_slope = lrelu_slope
        self.bias = use_bias
        self.LayerNorm = Layer_norm_process(self.features)
        self.conv1 = nn.Conv2d(self.features, self.features, kernel_size=(3, 3), stride=1, bias=self.bias, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=self.lrelu_slope)
        self.conv2 = nn.Conv2d(self.features, self.features, kernel_size=(3, 3), stride=1, bias=self.bias, padding=1)
        self.channel_attention = CALayer(features=self.features, reduction=self.reduction)

    def forward(self, x):
        shortcut = x
        x = self.LayerNorm(x)
        x = x.permute(0, 3, 1, 2)  # n, c, h, w
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)  # n, h, w, c
        x = self.channel_attention(x)
        return x + shortcut


class RDCAB(nn.Module):  # input shape: n, h, w, c
    """Residual dense channel attention block. Used in Bottlenecks."""

    def __init__(self, features, reduction=4, dropout_rate=0, use_bias=True):
        super().__init__()
        self.features = features
        self.reduction = reduction
        self.drop = dropout_rate
        self.bias = use_bias
        self.LayerNorm = Layer_norm_process(self.features)
        self.channel_mixing = MlpBlock(mlp_dim=self.features, dropout_rate=self.drop, use_bias=self.bias)
        self.channel_attention = CALayer(features=self.features, reduction=self.reduction, use_bias=self.bias)

    def forward(self, x):
        y = self.LayerNorm(x)
        y = self.channel_mixing(y)
        y = self.channel_attention(y)
        x = x + y
        return x


class CrossGatingBlock(nn.Module):  # input shape: n, c, h, w
    """Cross-gating MLP block."""

    def __init__(self, x_features, num_channels, block_size, grid_size, cin_y=0, upsample_y=True, use_bias=True,
                 use_global_mlp=True, dropout_rate=0):
        super().__init__()
        self.cin_y = cin_y
        self.x_features = x_features
        self.num_channels = num_channels
        self.block_size = block_size
        self.grid_size = grid_size
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        self.use_global_mlp = use_global_mlp
        self.drop = dropout_rate
        self.ConvTranspose_0 = nn.ConvTranspose2d(self.cin_y, self.num_channels, kernel_size=(2, 2), stride=2,
                                                  bias=self.use_bias)
        self.Conv_0 = nn.Conv2d(self.x_features, self.num_channels, kernel_size=(1, 1), stride=1, bias=self.use_bias)
        self.Conv_1 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=(1, 1), stride=1, bias=self.use_bias)
        self.LayerNorm_x = Layer_norm_process(self.num_channels)
        self.in_project_x = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.gelu1 = nn.GELU(approximate='tanh')
        self.SplitHeadMultiAxisGating_x = GetSpatialGatingWeights(num_channels=self.num_channels,
                                                                  block_size=self.block_size, grid_size=self.grid_size,
                                                                  dropout_rate=self.drop, use_bias=self.use_bias)
        self.LayerNorm_y = Layer_norm_process(self.num_channels)
        self.in_project_y = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.gelu2 = nn.GELU(approximate='tanh')
        self.SplitHeadMultiAxisGating_y = GetSpatialGatingWeights(num_channels=self.num_channels,
                                                                  block_size=self.block_size, grid_size=self.grid_size,
                                                                  dropout_rate=self.drop, use_bias=self.use_bias)
        self.out_project_y = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.dropout1 = nn.Dropout(self.drop)
        self.out_project_x = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.dropout2 = nn.Dropout(self.drop)

    def forward(self, x, y):
        # Upscale Y signal, y is the gating signal.
        if self.upsample_y:
            y = self.ConvTranspose_0(y)
        x = self.Conv_0(x)
        y = self.Conv_1(y)
        assert y.shape == x.shape
        x = x.permute(0, 2, 3, 1)  # n,h,w,c
        y = y.permute(0, 2, 3, 1)  # n,h,w,c
        shortcut_x = x
        shortcut_y = y
        # Get gating weights from X
        x = self.LayerNorm_x(x)
        x = self.in_project_x(x)
        x = self.gelu1(x)
        gx = self.SplitHeadMultiAxisGating_x(x)
        # Get gating weights from Y
        y = self.LayerNorm_y(y)
        y = self.in_project_y(y)
        y = self.gelu2(y)
        gy = self.SplitHeadMultiAxisGating_y(y)
        # Apply cross gating
        y = y * gx  ## gating y using x
        y = self.out_project_y(y)
        y = self.dropout1(y)
        y = y + shortcut_y
        x = x * gy  # gating x using y
        x = self.out_project_x(x)
        x = self.dropout2(x)
        x = x + y + shortcut_x  # get all aggregated signals
        return x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)  # n,c,h,w


class UNetEncoderBlock(nn.Module):  # input shape: n, c, h, w (pytorch default)
    """Encoder block in MAXIM."""

    def __init__(self, cin, num_channels, block_size, grid_size, dec=False, lrelu_slope=0.2, block_gmlp_factor=2,
                 grid_gmlp_factor=2,
                 input_proj_factor=2, channels_reduction=4, dropout_rate=0., use_bias=True, downsample=True,
                 use_global_mlp=True):
        super().__init__()
        self.cin = cin
        self.num_channels = num_channels
        self.block_size = block_size
        self.grid_size = grid_size
        self.lrelu_slope = lrelu_slope
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.reduction = channels_reduction
        self.drop = dropout_rate
        self.dec = dec
        self.use_bias = use_bias
        self.downsample = downsample
        self.use_global_mlp = use_global_mlp
        self.Conv_0 = nn.Conv2d(self.cin, self.num_channels, kernel_size=(1, 1), stride=(1, 1), bias=self.use_bias)
        self.SplitHeadMultiAxisGmlpLayer_0 = ResidualSplitHeadMultiAxisGmlpLayer(block_size=self.block_size,
                                                                                 grid_size=self.grid_size,
                                                                                 num_channels=self.num_channels,
                                                                                 input_proj_factor=self.input_proj_factor,
                                                                                 block_gmlp_factor=self.block_gmlp_factor,
                                                                                 grid_gmlp_factor=self.grid_gmlp_factor,
                                                                                 dropout_rate=self.drop,
                                                                                 use_bias=self.use_bias)
        self.SplitHeadMultiAxisGmlpLayer_1 = ResidualSplitHeadMultiAxisGmlpLayer(block_size=self.block_size,
                                                                                 grid_size=self.grid_size,
                                                                                 num_channels=self.num_channels,
                                                                                 input_proj_factor=self.input_proj_factor,
                                                                                 block_gmlp_factor=self.block_gmlp_factor,
                                                                                 grid_gmlp_factor=self.grid_gmlp_factor,
                                                                                 dropout_rate=self.drop,
                                                                                 use_bias=self.use_bias)
        self.channel_attention_block_10 = RCAB(features=self.num_channels, reduction=self.reduction,
                                               lrelu_slope=self.lrelu_slope, use_bias=self.use_bias)
        self.channel_attention_block_11 = RCAB(features=self.num_channels, reduction=self.reduction,
                                               lrelu_slope=self.lrelu_slope, use_bias=self.use_bias)
        self.cross_gating_block = CrossGatingBlock(x_features=self.num_channels, num_channels=self.num_channels,
                                                   block_size=self.block_size,
                                                   grid_size=self.grid_size, upsample_y=False, dropout_rate=self.drop,
                                                   use_bias=self.use_bias, use_global_mlp=self.use_global_mlp)
        self.Conv_1 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, x, skip=None, enc=None, dec=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.Conv_0(x)
        shortcut_long = x
        x = x.permute(0, 2, 3, 1)  # n,h,w,c
        if self.use_global_mlp:
            x = self.SplitHeadMultiAxisGmlpLayer_0(x)
        x = self.channel_attention_block_10(x)
        if self.use_global_mlp:
            x = self.SplitHeadMultiAxisGmlpLayer_1(x)
        x = self.channel_attention_block_11(x)
        x = x.permute(0, 3, 1, 2)  # n,c,h,w
        x = x + shortcut_long
        if enc is not None and dec is not None:  # if stage>0
            x, _ = self.cross_gating_block(x, enc + dec)
        if self.downsample:
            x_down = self.Conv_1(x)
            return x_down, x
        else:
            return x


class UNetDecoderBlock(nn.Module):  # input shape: n, c, h, w
    """Decoder block in MAXIM."""

    def __init__(self, cin, num_channels, block_size, grid_size, lrelu_slope=0.2, block_gmlp_factor=2,
                 grid_gmlp_factor=2,
                 input_proj_factor=2, channels_reduction=4, dropout_rate=0., use_bias=True, downsample=True,
                 use_global_mlp=True):
        super().__init__()
        self.cin = cin
        self.num_channels = num_channels
        self.block_size = block_size
        self.grid_size = grid_size
        self.lrelu_slope = lrelu_slope
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.reduction = channels_reduction
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.downsample = downsample
        self.use_global_mlp = use_global_mlp
        self.ConvTranspose_0 = nn.ConvTranspose2d(self.cin, self.num_channels, kernel_size=(2, 2), stride=2,
                                                  bias=self.use_bias)
        self.UNetEncoderBlock_0 = UNetEncoderBlock(4 * self.num_channels, self.num_channels, self.block_size,
                                                   self.grid_size, lrelu_slope=self.lrelu_slope,
                                                   block_gmlp_factor=self.block_gmlp_factor,
                                                   grid_gmlp_factor=self.grid_gmlp_factor, dec=True,
                                                   input_proj_factor=self.input_proj_factor,
                                                   channels_reduction=self.reduction, dropout_rate=self.dropout_rate,
                                                   use_bias=self.use_bias, downsample=False,
                                                   use_global_mlp=self.use_global_mlp)

    def forward(self, x, bridge=None):
        x = self.ConvTranspose_0(x)
        x = self.UNetEncoderBlock_0(x, skip=bridge)
        return x


class BottleneckBlock(nn.Module):  # input shape: n,c,h,w
    """The bottleneck block consisting of multi-axis gMLP block and RDCAB."""

    def __init__(self, features, block_size, grid_size, block_gmlp_factor=2, grid_gmlp_factor=2, input_proj_factor=2,
                 channels_reduction=4, use_bias=True, dropout_rate=0.):
        super().__init__()
        self.features = features
        self.block_size = block_size
        self.grid_size = grid_size
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.channels_reduction = channels_reduction
        self.use_bias = use_bias
        self.drop = dropout_rate
        self.input_proj = nn.Conv2d(self.features, self.features, kernel_size=(1, 1), stride=1)
        self.SplitHeadMultiAxisGmlpLayer_0 = ResidualSplitHeadMultiAxisGmlpLayer(block_size=self.block_size,
                                                                                 grid_size=self.grid_size,
                                                                                 num_channels=self.features,
                                                                                 input_proj_factor=self.input_proj_factor,
                                                                                 block_gmlp_factor=self.block_gmlp_factor,
                                                                                 grid_gmlp_factor=self.grid_gmlp_factor,
                                                                                 use_bias=self.use_bias)
        self.SplitHeadMultiAxisGmlpLayer_1 = ResidualSplitHeadMultiAxisGmlpLayer(block_size=self.block_size,
                                                                                 grid_size=self.grid_size,
                                                                                 num_channels=self.features,
                                                                                 input_proj_factor=self.input_proj_factor,
                                                                                 block_gmlp_factor=self.block_gmlp_factor,
                                                                                 grid_gmlp_factor=self.grid_gmlp_factor,
                                                                                 use_bias=self.use_bias)
        self.channel_attention_block_1_0 = RDCAB(features, dropout_rate=self.drop, use_bias=self.use_bias)
        self.channel_attention_block_1_1 = RDCAB(features, dropout_rate=self.drop, use_bias=self.use_bias)

    def forward(self, x):
        assert x.ndim == 4  # Input has shape [batch, c, h, w]
        # input projection
        x = self.input_proj(x)
        shortcut_long = x
        x = x.permute(0, 2, 3, 1)  # n, h, w, c
        x = self.SplitHeadMultiAxisGmlpLayer_0(x)
        x = self.channel_attention_block_1_0(x)
        x = self.SplitHeadMultiAxisGmlpLayer_1(x)
        x = self.channel_attention_block_1_1(x)
        x = x.permute(0, 3, 1, 2)  # n, c, h, w
        x = x + shortcut_long
        return x


# multi stage
class SAM(nn.Module):  # x shape and x_image shape: n, c, h, w
    """Supervised attention module for multi-stage training.
    Introduced by MPRNet [CVPR2021]: https://github.com/swz30/MPRNet
    """

    def __init__(self, features, output_channels=3, use_bias=True):
        super().__init__()
        self.features = features  # cin
        self.output_channels = output_channels
        self.use_bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, self.features, kernel_size=(3, 3), bias=self.use_bias, padding=1)
        self.Conv_1 = nn.Conv2d(self.features, self.output_channels, kernel_size=(3, 3), bias=self.use_bias, padding=1)
        self.Conv_2 = nn.Conv2d(self.output_channels, self.features, kernel_size=(3, 3), bias=self.use_bias, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_image):
        """Apply the SAM module to the input and features.
        Args:
          x: the output features from UNet decoder with shape (h, w, c)
          x_image: the input image with shape (h, w, 3)
          train: Whether it is training
        Returns:
          A tuple of tensors (x1, image) where (x1) is the sam features used for the
            next stage, and (image) is the output restored image at current stage.
        """
        # Get features
        x1 = self.Conv_0(x)
        # Output restored image X_s
        if self.output_channels == 3:
            image = self.Conv_1(x) + x_image
        else:
            image = self.Conv_1(x)
        # Get attention maps for features
        x2 = self.Conv_2(image)
        x2 = self.sigmoid(x2)
        # Get attended feature maps
        x1 = x1 * x2
        # Residual connection
        x1 = x1 + x
        return x1, image


# top: 3-stage MAXIM for image denoising
class MAXIM_dns_3s(nn.Module):  # input shape: n, c, h, w
    """The MAXIM model function with multi-stage and multi-scale supervision.
    For more model details, please check the CVPR paper:
    MAXIM: MUlti-Axis MLP for Image Processing (https://arxiv.org/abs/2201.02973)
    Attributes:
      features: initial hidden dimension for the input resolution.
      depth: the number of downsampling depth for the model.
      num_stages: how many stages to use. It will also affects the output list.
      use_bias: whether to use bias in all the conv/mlp layers.
      num_supervision_scales: the number of desired supervision scales.
      lrelu_slope: the negative slope parameter in leaky_relu layers.
      use_global_mlp: whether to use the multi-axis gated MLP block (MAB) in each
        layer.
      use_cross_gating: whether to use the cross-gating MLP block (CGB) in the
        skip connections and multi-stage feature fusion layers.
      high_res_stages: how many stages are specificied as high-res stages. The
        rest (depth - high_res_stages) are called low_res_stages.
      block_size_hr: the block_size parameter for high-res stages.
      block_size_lr: the block_size parameter for low-res stages.
      grid_size_hr: the grid_size parameter for high-res stages.
      grid_size_lr: the grid_size parameter for low-res stages.
      block_gmlp_factor: the input projection factor for block_gMLP layers.
      grid_gmlp_factor: the input projection factor for grid_gMLP layers.
      input_proj_factor: the input projection factor for the MAB block.
      channels_reduction: the channel reduction factor for SE layer.
      num_outputs: the output channels.
      dropout_rate: Dropout rate.
    Returns:
      The output contains a list of arrays consisting of multi-stage multi-scale
      outputs. For example, if num_stages = num_supervision_scales = 3 (the
      model used in the paper), the output specs are: outputs =
      [[output_stage1_scale1, output_stage1_scale2, output_stage1_scale3],
       [output_stage2_scale1, output_stage2_scale2, output_stage2_scale3],
       [output_stage3_scale1, output_stage3_scale2, output_stage3_scale3],]
      The final output can be retrieved by outputs[-1][-1].
    """

    def __init__(self, features=32, depth=3, use_bias=True, num_supervision_scales=int(3), lrelu_slope=0.2,
                 use_global_mlp=True, high_res_stages=2, block_size_hr=(16, 16), block_size_lr=(8, 8),
                 grid_size_hr=(16, 16), grid_size_lr=(8, 8),
                 block_gmlp_factor=2, grid_gmlp_factor=2, input_proj_factor=2, channels_reduction=4, num_outputs=3,
                 dropout_rate=0.):
        super().__init__()
        self.features = features
        self.depth = depth
        self.bias = use_bias
        self.num_supervision_scales = num_supervision_scales
        self.lrelu_slope = lrelu_slope
        self.use_global_mlp = use_global_mlp
        self.high_res_stages = high_res_stages
        self.block_size_hr = block_size_hr
        self.block_size_lr = block_size_lr
        self.grid_size_hr = grid_size_hr
        self.grid_size_lr = grid_size_lr
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.channels_reduction = channels_reduction
        self.num_outputs = num_outputs
        self.drop = dropout_rate

        ########## STAGE 0 ##########
        # multi scale input and encoder
        # depth=0
        self.stage_0_input_conv_0 = nn.Conv2d(3, self.features, kernel_size=(3, 3), bias=self.bias, padding=1)
        self.stage_0_encoder_block_0 = UNetEncoderBlock(cin=2 * self.features, num_channels=self.features,
                                                        block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        dropout_rate=self.drop, use_bias=self.bias,
                                                        downsample=True, use_global_mlp=self.use_global_mlp)
        # depth=1
        self.stage_0_input_conv_1 = nn.Conv2d(3, 2 * self.features, kernel_size=(3, 3), bias=self.bias, padding=1)
        self.stage_0_encoder_block_1 = UNetEncoderBlock(cin=3 * self.features, num_channels=2 * self.features,
                                                        block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        dropout_rate=self.drop, use_bias=self.bias,
                                                        downsample=True, use_global_mlp=self.use_global_mlp)
        # depth=2
        self.stage_0_input_conv_2 = nn.Conv2d(3, 4 * self.features, kernel_size=(3, 3), bias=self.bias, padding=1)
        self.stage_0_encoder_block_2 = UNetEncoderBlock(cin=6 * self.features, num_channels=4 * self.features,
                                                        block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        dropout_rate=self.drop, use_bias=self.bias,
                                                        downsample=True, use_global_mlp=self.use_global_mlp)

        # bottleneck
        self.stage_0_global_block_0 = BottleneckBlock(block_size=self.block_size_lr, grid_size=self.grid_size_lr,
                                                      features=4 * self.features,
                                                      block_gmlp_factor=self.block_gmlp_factor,
                                                      grid_gmlp_factor=self.grid_gmlp_factor,
                                                      input_proj_factor=self.input_proj_factor,
                                                      dropout_rate=self.drop, use_bias=self.bias,
                                                      channels_reduction=self.channels_reduction)
        self.stage_0_global_block_1 = BottleneckBlock(block_size=self.block_size_lr, grid_size=self.grid_size_lr,
                                                      features=4 * self.features,
                                                      block_gmlp_factor=self.block_gmlp_factor,
                                                      grid_gmlp_factor=self.grid_gmlp_factor,
                                                      input_proj_factor=self.input_proj_factor,
                                                      dropout_rate=self.drop, use_bias=self.bias,
                                                      channels_reduction=self.channels_reduction)

        # cross gating (within a stage)
        # depth=2
        self.UpSampleRatio_0 = UpSampleRatio_4(1 * self.features, ratio=2 ** (-2), use_bias=self.bias)  # 0->2
        self.UpSampleRatio_1 = UpSampleRatio_2(2 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 1->2
        self.UpSampleRatio_2 = UpSampleRatio(4 * self.features, ratio=1, use_bias=self.bias)  # 2->2
        self.stage_0_cross_gating_block_2 = CrossGatingBlock(cin_y=4 * features,
                                                             x_features=3 * (2 ** 2) * self.features,
                                                             num_channels=4 * features,
                                                             block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                             grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                             upsample_y=True, use_bias=self.bias,
                                                             dropout_rate=self.drop)
        # depth=1
        self.UpSampleRatio_3 = UpSampleRatio_2(1 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 0->1
        self.UpSampleRatio_4 = UpSampleRatio(2 * self.features, ratio=2 ** (0), use_bias=self.bias)  # 1->1
        self.UpSampleRatio_5 = UpSampleRatio_1_2(4 * self.features, ratio=2, use_bias=self.bias)  # 2->1
        self.stage_0_cross_gating_block_1 = CrossGatingBlock(cin_y=4 * features, x_features=3 * 2 * self.features,
                                                             num_channels=2 * features,
                                                             block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                             grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                             upsample_y=True, use_bias=self.bias,
                                                             dropout_rate=self.drop)
        # depth=0
        self.UpSampleRatio_6 = UpSampleRatio(1 * self.features, ratio=1, use_bias=self.bias)  # 0->0
        self.UpSampleRatio_7 = UpSampleRatio_1_2(2 * self.features, ratio=2, use_bias=self.bias)  # 1->0
        self.UpSampleRatio_8 = UpSampleRatio_1_4(4 * self.features, ratio=4, use_bias=self.bias)  # 2->0
        self.stage_0_cross_gating_block_0 = CrossGatingBlock(cin_y=2 * features, x_features=3 * self.features,
                                                             num_channels=self.features,
                                                             block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                             grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                             upsample_y=True, use_bias=self.bias,
                                                             dropout_rate=self.drop)

        # decoder
        # depth=2
        self.UpSampleRatio_9 = UpSampleRatio(4 * self.features, ratio=2 ** (0), use_bias=self.bias)  # 2->2
        self.UpSampleRatio_10 = UpSampleRatio_2(2 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 1->2
        self.UpSampleRatio_11 = UpSampleRatio_4(1 * self.features, ratio=2 ** (-2), use_bias=self.bias)  # 0->2
        self.stage_0_decoder_block_2 = UNetDecoderBlock(cin=4 * self.features, num_channels=(2 ** 2) * self.features,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        use_global_mlp=self.use_global_mlp, dropout_rate=self.drop,
                                                        use_bias=self.bias)
        self.stage_0_supervised_attention_module_2 = SAM(features=2 ** (2) * self.features,
                                                         output_channels=self.num_outputs, use_bias=self.bias)
        # depth=1
        self.UpSampleRatio_12 = UpSampleRatio_1_2(4 * self.features, ratio=2 ** (1), use_bias=self.bias)  # 2->1
        self.UpSampleRatio_13 = UpSampleRatio(2 * self.features, ratio=2 ** (0), use_bias=self.bias)  # 1->1
        self.UpSampleRatio_14 = UpSampleRatio_2(1 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 0->1
        self.stage_0_decoder_block_1 = UNetDecoderBlock(cin=4 * self.features, num_channels=(2 ** 1) * self.features,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        use_global_mlp=self.use_global_mlp, dropout_rate=self.drop,
                                                        use_bias=self.bias)
        self.stage_0_supervised_attention_module_1 = SAM(features=2 ** (1) * self.features,
                                                         output_channels=self.num_outputs, use_bias=self.bias)
        # depth=0
        self.UpSampleRatio_15 = UpSampleRatio_1_4(4 * self.features, ratio=4, use_bias=self.bias)  # 2->0
        self.UpSampleRatio_16 = UpSampleRatio_1_2(2 * self.features, ratio=2, use_bias=self.bias)  # 1->0
        self.UpSampleRatio_17 = UpSampleRatio(1 * self.features, ratio=1, use_bias=self.bias)  # 0->0
        self.stage_0_decoder_block_0 = UNetDecoderBlock(cin=2 * self.features, num_channels=self.features,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        use_global_mlp=self.use_global_mlp, dropout_rate=self.drop,
                                                        use_bias=self.bias)
        self.stage_0_supervised_attention_module_0 = SAM(features=2 ** (0) * self.features,
                                                         output_channels=self.num_outputs, use_bias=self.bias)

        ########## STAGE 1 ##########
        # multi scale input and encoder
        # depth=0
        self.stage_1_input_conv_0 = nn.Conv2d(3, self.features, kernel_size=(3, 3), bias=self.bias, padding=1)
        self.stage_1_input_fuse_sam_0 = CrossGatingBlock(x_features=self.features, num_channels=self.features,
                                                         block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                         grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                         upsample_y=False, use_bias=self.bias, dropout_rate=self.drop)
        self.stage_1_input_catconv_0 = nn.Conv2d(2 * self.features, self.features, kernel_size=(1, 1), bias=self.bias)
        self.stage_1_encoder_block_0 = UNetEncoderBlock(cin=2 * self.features, num_channels=self.features,
                                                        block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        dropout_rate=self.drop, use_bias=self.bias,
                                                        downsample=True, use_global_mlp=self.use_global_mlp)
        # depth=1
        self.stage_1_input_conv_1 = nn.Conv2d(3, 2 * self.features, kernel_size=(3, 3), bias=self.bias, padding=1)
        self.stage_1_input_fuse_sam_1 = CrossGatingBlock(x_features=2 * self.features, num_channels=2 * self.features,
                                                         block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                         grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                         upsample_y=False, use_bias=self.bias, dropout_rate=self.drop)
        self.stage_1_input_catconv_1 = nn.Conv2d(4 * self.features, 2 * self.features, kernel_size=(1, 1),
                                                 bias=self.bias)
        self.stage_1_encoder_block_1 = UNetEncoderBlock(cin=3 * self.features, num_channels=2 * self.features,
                                                        block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        dropout_rate=self.drop, use_bias=self.bias,
                                                        downsample=True, use_global_mlp=self.use_global_mlp)
        # depth=2
        self.stage_1_input_conv_2 = nn.Conv2d(3, 4 * self.features, kernel_size=(3, 3), bias=self.bias, padding=1)
        self.stage_1_input_fuse_sam_2 = CrossGatingBlock(x_features=4 * self.features, num_channels=4 * self.features,
                                                         block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                         grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                         upsample_y=False, use_bias=self.bias, dropout_rate=self.drop)
        self.stage_1_input_catconv_2 = nn.Conv2d(4 * self.features, 4 * self.features, kernel_size=(1, 1),
                                                 bias=self.bias)
        self.stage_1_encoder_block_2 = UNetEncoderBlock(cin=6 * self.features, num_channels=4 * self.features,
                                                        block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        dropout_rate=self.drop, use_bias=self.bias,
                                                        downsample=True, use_global_mlp=self.use_global_mlp)

        # bottleneck
        self.stage_1_global_block_0 = BottleneckBlock(block_size=self.block_size_lr, grid_size=self.grid_size_lr,
                                                      features=4 * self.features,
                                                      block_gmlp_factor=self.block_gmlp_factor,
                                                      grid_gmlp_factor=self.grid_gmlp_factor,
                                                      input_proj_factor=self.input_proj_factor,
                                                      dropout_rate=self.drop, use_bias=self.bias,
                                                      channels_reduction=self.channels_reduction)
        self.stage_1_global_block_1 = BottleneckBlock(block_size=self.block_size_lr, grid_size=self.grid_size_lr,
                                                      features=4 * self.features,
                                                      block_gmlp_factor=self.block_gmlp_factor,
                                                      grid_gmlp_factor=self.grid_gmlp_factor,
                                                      input_proj_factor=self.input_proj_factor,
                                                      dropout_rate=self.drop, use_bias=self.bias,
                                                      channels_reduction=self.channels_reduction)

        # cross gating
        # depth=2
        self.UpSampleRatio_18 = UpSampleRatio_4(1 * self.features, ratio=2 ** (-2), use_bias=self.bias)  # 0->2
        self.UpSampleRatio_19 = UpSampleRatio_2(2 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 1->2
        self.UpSampleRatio_20 = UpSampleRatio(4 * self.features, ratio=1, use_bias=self.bias)  # 2->2
        self.stage_1_cross_gating_block_2 = CrossGatingBlock(cin_y=4 * features,
                                                             x_features=3 * (2 ** 2) * self.features,
                                                             num_channels=4 * features,
                                                             block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                             grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                             upsample_y=True, use_bias=self.bias,
                                                             dropout_rate=self.drop)
        # depth=1
        self.UpSampleRatio_21 = UpSampleRatio_2(1 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 0->1
        self.UpSampleRatio_22 = UpSampleRatio(2 * self.features, ratio=2 ** (0), use_bias=self.bias)  # 1->1
        self.UpSampleRatio_23 = UpSampleRatio_1_2(4 * self.features, ratio=2, use_bias=self.bias)  # 2->1
        self.stage_1_cross_gating_block_1 = CrossGatingBlock(cin_y=4 * features, x_features=3 * 2 * self.features,
                                                             num_channels=2 * features,
                                                             block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                             grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                             upsample_y=True, use_bias=self.bias,
                                                             dropout_rate=self.drop)
        # depth=0
        self.UpSampleRatio_24 = UpSampleRatio(1 * self.features, ratio=1, use_bias=self.bias)  # 0->0
        self.UpSampleRatio_25 = UpSampleRatio_1_2(2 * self.features, ratio=2, use_bias=self.bias)  # 1->0
        self.UpSampleRatio_26 = UpSampleRatio_1_4(4 * self.features, ratio=4, use_bias=self.bias)  # 2->0
        self.stage_1_cross_gating_block_0 = CrossGatingBlock(cin_y=2 * features, x_features=3 * self.features,
                                                             num_channels=self.features,
                                                             block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                             grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                             upsample_y=True, use_bias=self.bias,
                                                             dropout_rate=self.drop)

        # decoder
        # depth=2
        self.UpSampleRatio_27 = UpSampleRatio(4 * self.features, ratio=2 ** (0), use_bias=self.bias)  # 2->2
        self.UpSampleRatio_28 = UpSampleRatio_2(2 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 1->2
        self.UpSampleRatio_29 = UpSampleRatio_4(1 * self.features, ratio=2 ** (-2), use_bias=self.bias)  # 0->2
        self.stage_1_decoder_block_2 = UNetDecoderBlock(cin=4 * self.features, num_channels=(2 ** 2) * self.features,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        use_global_mlp=self.use_global_mlp, dropout_rate=self.drop,
                                                        use_bias=self.bias)
        self.stage_1_supervised_attention_module_2 = SAM(features=2 ** (2) * self.features,
                                                         output_channels=self.num_outputs, use_bias=self.bias)
        # depth=1
        self.UpSampleRatio_30 = UpSampleRatio_1_2(4 * self.features, ratio=2 ** (1), use_bias=self.bias)  # 2->1
        self.UpSampleRatio_31 = UpSampleRatio(2 * self.features, ratio=2 ** (0), use_bias=self.bias)  # 1->1
        self.UpSampleRatio_32 = UpSampleRatio_2(1 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 0->1
        self.stage_1_decoder_block_1 = UNetDecoderBlock(cin=4 * self.features, num_channels=(2 ** 1) * self.features,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        use_global_mlp=self.use_global_mlp, dropout_rate=self.drop,
                                                        use_bias=self.bias)
        self.stage_1_supervised_attention_module_1 = SAM(features=2 ** (1) * self.features,
                                                         output_channels=self.num_outputs, use_bias=self.bias)
        # depth=0
        self.UpSampleRatio_33 = UpSampleRatio_1_4(4 * self.features, ratio=4, use_bias=self.bias)  # 2->0
        self.UpSampleRatio_34 = UpSampleRatio_1_2(2 * self.features, ratio=2, use_bias=self.bias)  # 1->0
        self.UpSampleRatio_35 = UpSampleRatio(1 * self.features, ratio=1, use_bias=self.bias)  # 0->0
        self.stage_1_decoder_block_0 = UNetDecoderBlock(cin=2 * self.features, num_channels=self.features,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        use_global_mlp=self.use_global_mlp, dropout_rate=self.drop,
                                                        use_bias=self.bias)
        self.stage_1_supervised_attention_module_0 = SAM(features=2 ** (0) * self.features,
                                                         output_channels=self.num_outputs, use_bias=self.bias)

        ########## STAGE 2 ##########
        # multi scale input and encoder
        # depth=0
        self.stage_2_input_conv_0 = nn.Conv2d(3, self.features, kernel_size=(3, 3), bias=self.bias, padding=1)
        self.stage_2_input_fuse_sam_0 = CrossGatingBlock(x_features=self.features, num_channels=self.features,
                                                         block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                         grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                         upsample_y=False, use_bias=self.bias, dropout_rate=self.drop)
        self.stage_2_input_catconv_0 = nn.Conv2d(2 * self.features, self.features, kernel_size=(1, 1), bias=self.bias)
        self.stage_2_encoder_block_0 = UNetEncoderBlock(cin=2 * self.features, num_channels=self.features,
                                                        block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        dropout_rate=self.drop, use_bias=self.bias,
                                                        downsample=True, use_global_mlp=self.use_global_mlp)
        # depth=1
        self.stage_2_input_conv_1 = nn.Conv2d(3, 2 * self.features, kernel_size=(3, 3), bias=self.bias, padding=1)
        self.stage_2_input_fuse_sam_1 = CrossGatingBlock(x_features=2 * self.features, num_channels=2 * self.features,
                                                         block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                         grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                         upsample_y=False, use_bias=self.bias, dropout_rate=self.drop)
        self.stage_2_input_catconv_1 = nn.Conv2d(4 * self.features, 2 * self.features, kernel_size=(1, 1),
                                                 bias=self.bias)
        self.stage_2_encoder_block_1 = UNetEncoderBlock(cin=3 * self.features, num_channels=2 * self.features,
                                                        block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        dropout_rate=self.drop, use_bias=self.bias,
                                                        downsample=True, use_global_mlp=self.use_global_mlp)
        # depth=2
        self.stage_2_input_conv_2 = nn.Conv2d(3, 4 * self.features, kernel_size=(3, 3), bias=self.bias, padding=1)
        self.stage_2_input_fuse_sam_2 = CrossGatingBlock(x_features=4 * self.features, num_channels=4 * self.features,
                                                         block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                         grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                         upsample_y=False, use_bias=self.bias, dropout_rate=self.drop)
        self.stage_2_input_catconv_2 = nn.Conv2d(4 * self.features, 4 * self.features, kernel_size=(1, 1),
                                                 bias=self.bias)
        self.stage_2_encoder_block_2 = UNetEncoderBlock(cin=6 * self.features, num_channels=4 * self.features,
                                                        block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        dropout_rate=self.drop, use_bias=self.bias,
                                                        downsample=True, use_global_mlp=self.use_global_mlp)

        # bottleneck
        self.stage_2_global_block_0 = BottleneckBlock(block_size=self.block_size_lr, grid_size=self.grid_size_lr,
                                                      features=4 * self.features,
                                                      block_gmlp_factor=self.block_gmlp_factor,
                                                      grid_gmlp_factor=self.grid_gmlp_factor,
                                                      input_proj_factor=self.input_proj_factor,
                                                      dropout_rate=self.drop, use_bias=self.bias,
                                                      channels_reduction=self.channels_reduction)
        self.stage_2_global_block_1 = BottleneckBlock(block_size=self.block_size_lr, grid_size=self.grid_size_lr,
                                                      features=4 * self.features,
                                                      block_gmlp_factor=self.block_gmlp_factor,
                                                      grid_gmlp_factor=self.grid_gmlp_factor,
                                                      input_proj_factor=self.input_proj_factor,
                                                      dropout_rate=self.drop, use_bias=self.bias,
                                                      channels_reduction=self.channels_reduction)

        # cross gating
        # depth=2
        self.UpSampleRatio_36 = UpSampleRatio_4(1 * self.features, ratio=2 ** (-2), use_bias=self.bias)  # 0->2
        self.UpSampleRatio_37 = UpSampleRatio_2(2 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 1->2
        self.UpSampleRatio_38 = UpSampleRatio(4 * self.features, ratio=1, use_bias=self.bias)  # 2->2
        self.stage_2_cross_gating_block_2 = CrossGatingBlock(cin_y=4 * features,
                                                             x_features=3 * (2 ** 2) * self.features,
                                                             num_channels=4 * features,
                                                             block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                             grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                             upsample_y=True, use_bias=self.bias,
                                                             dropout_rate=self.drop)
        # depth=1
        self.UpSampleRatio_39 = UpSampleRatio_2(1 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 0->1
        self.UpSampleRatio_40 = UpSampleRatio(2 * self.features, ratio=2 ** (0), use_bias=self.bias)  # 1->1
        self.UpSampleRatio_41 = UpSampleRatio_1_2(4 * self.features, ratio=2, use_bias=self.bias)  # 2->1
        self.stage_2_cross_gating_block_1 = CrossGatingBlock(cin_y=4 * features, x_features=3 * 2 * self.features,
                                                             num_channels=2 * features,
                                                             block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                             grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                             upsample_y=True, use_bias=self.bias,
                                                             dropout_rate=self.drop)
        # depth=0
        self.UpSampleRatio_42 = UpSampleRatio(1 * self.features, ratio=1, use_bias=self.bias)  # 0->0
        self.UpSampleRatio_43 = UpSampleRatio_1_2(2 * self.features, ratio=2, use_bias=self.bias)  # 1->0
        self.UpSampleRatio_44 = UpSampleRatio_1_4(4 * self.features, ratio=4, use_bias=self.bias)  # 2->0
        self.stage_2_cross_gating_block_0 = CrossGatingBlock(cin_y=2 * features, x_features=3 * self.features,
                                                             num_channels=self.features,
                                                             block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                             grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                             upsample_y=True, use_bias=self.bias,
                                                             dropout_rate=self.drop)

        # decoder
        # depth=2
        self.UpSampleRatio_45 = UpSampleRatio(4 * self.features, ratio=2 ** (0), use_bias=self.bias)  # 2->2
        self.UpSampleRatio_46 = UpSampleRatio_2(2 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 1->2
        self.UpSampleRatio_47 = UpSampleRatio_4(1 * self.features, ratio=2 ** (-2), use_bias=self.bias)  # 0->2
        self.stage_2_decoder_block_2 = UNetDecoderBlock(cin=4 * self.features, num_channels=(2 ** 2) * self.features,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        use_global_mlp=self.use_global_mlp, dropout_rate=self.drop,
                                                        use_bias=self.bias)
        self.stage_2_output_conv_2 = nn.Conv2d((2 ** (2)) * self.features, self.num_outputs, kernel_size=(3, 3),
                                               bias=self.bias, padding=1)
        # depth=1
        self.UpSampleRatio_48 = UpSampleRatio_1_2(4 * self.features, ratio=2 ** (1), use_bias=self.bias)  # 2->1
        self.UpSampleRatio_49 = UpSampleRatio(2 * self.features, ratio=2 ** (0), use_bias=self.bias)  # 1->1
        self.UpSampleRatio_50 = UpSampleRatio_2(1 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 0->1
        self.stage_2_decoder_block_1 = UNetDecoderBlock(cin=4 * self.features, num_channels=(2 ** 1) * self.features,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        use_global_mlp=self.use_global_mlp, dropout_rate=self.drop,
                                                        use_bias=self.bias)
        self.stage_2_output_conv_1 = nn.Conv2d((2 ** (1)) * self.features, self.num_outputs, kernel_size=(3, 3),
                                               bias=self.bias, padding=1)
        # depth=0
        self.UpSampleRatio_51 = UpSampleRatio_1_4(4 * self.features, ratio=4, use_bias=self.bias)  # 2->0
        self.UpSampleRatio_52 = UpSampleRatio_1_2(2 * self.features, ratio=2, use_bias=self.bias)  # 1->0
        self.UpSampleRatio_53 = UpSampleRatio(1 * self.features, ratio=1, use_bias=self.bias)  # 0->0
        self.stage_2_decoder_block_0 = UNetDecoderBlock(cin=2 * self.features, num_channels=self.features,
                                                        lrelu_slope=self.lrelu_slope,
                                                        block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                        grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                        block_gmlp_factor=self.block_gmlp_factor,
                                                        grid_gmlp_factor=self.grid_gmlp_factor,
                                                        input_proj_factor=self.input_proj_factor,
                                                        channels_reduction=self.channels_reduction,
                                                        use_global_mlp=self.use_global_mlp, dropout_rate=self.drop,
                                                        use_bias=self.bias)
        self.stage_2_output_conv_0 = nn.Conv2d((2 ** (0)) * self.features, self.num_outputs, kernel_size=(3, 3),
                                               bias=self.bias, padding=1)

    def forward(self, x):
        shortcuts = []
        shortcuts.append(x)  # to store multiscale input images
        # Get multi-scale input images
        for i in range(1, self.num_supervision_scales):
            shortcuts.append(nearest_downsample(x, 1. / (2 ** i)))

        # store outputs from all stages and all scales
        # Eg, [[(64, 64, 3), (128, 128, 3), (256, 256, 3)],   # Stage-1 outputs
        #      [(64, 64, 3), (128, 128, 3), (256, 256, 3)],]  # Stage-2 outputs
        outputs_all = []
        sam_features, encs_prev, decs_prev = [], [], []  # to next stage

        # different stages
        ########## STAGE 0 ##########
        # Input convolution, get multi-scale input features
        x_scales = []
        for i in range(self.num_supervision_scales):
            if i == 0:
                x_scale = self.stage_0_input_conv_0(shortcuts[i])
                x_scales.append(x_scale)
            elif i == 1:
                x_scale = self.stage_0_input_conv_1(shortcuts[i])
                x_scales.append(x_scale)
            else:
                x_scale = self.stage_0_input_conv_2(shortcuts[i])
                x_scales.append(x_scale)
        # start encoder blocks
        encs = []
        x = x_scales[0]  # First full-scale input feature
        # use larger blocksize at high-res stages, vice versa.
        for i in range(self.depth):
            x_scale = x_scales[i] if i < self.num_supervision_scales else None
            enc_prev = None
            dec_prev = None
            if i == 0:
                x, bridge = self.stage_0_encoder_block_0(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
            elif i == 1:
                x, bridge = self.stage_0_encoder_block_1(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
            else:
                x, bridge = self.stage_0_encoder_block_2(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
            encs.append(bridge)
        # Global MLP bottleneck blocks
        x = self.stage_0_global_block_0(x)
        x = self.stage_0_global_block_1(x)

        # cache global feature for cross-gating
        global_feature = x

        # start cross gating. Use multi-scale feature fusion
        skip_features = []
        for i in reversed(range(self.depth)):  # 2, 1, 0
            if i == 2:
                # get multi-scale skip signals from cross-gating block
                signal0 = self.UpSampleRatio_0(encs[0])
                signal1 = self.UpSampleRatio_1(encs[1])
                signal2 = self.UpSampleRatio_2(encs[2])
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                # Use cross-gating to cross modulate features
                skips, global_feature = self.stage_0_cross_gating_block_2(signal, global_feature)
                skip_features.append(skips)
            elif i == 1:
                # get multi-scale skip signals from cross-gating block
                signal0 = self.UpSampleRatio_3(encs[0])
                signal1 = self.UpSampleRatio_4(encs[1])
                signal2 = self.UpSampleRatio_5(encs[2])
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                # Use cross-gating to cross modulate features
                skips, global_feature = self.stage_0_cross_gating_block_1(signal, global_feature)
                skip_features.append(skips)
            elif i == 0:
                # get multi-scale skip signals from cross-gating block
                signal0 = self.UpSampleRatio_6(encs[0])
                signal1 = self.UpSampleRatio_7(encs[1])
                signal2 = self.UpSampleRatio_8(encs[2])
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                # Use cross-gating to cross modulate features
                skips, global_feature = self.stage_0_cross_gating_block_0(signal, global_feature)
                skip_features.append(skips)

        # start decoder. Multi-scale feature fusion of cross-gated features
        outputs, decs, sam_features = [], [], []
        for i in reversed(range(self.depth)):
            if i == 2:
                # get multi-scale skip signals from cross-gating block
                signal2 = self.UpSampleRatio_9(skip_features[0])
                signal1 = self.UpSampleRatio_10(skip_features[1])
                signal0 = self.UpSampleRatio_11(skip_features[2])
                signal = torch.cat([signal2, signal1, signal0], dim=1)
                # Decoder block
                x = self.stage_0_decoder_block_2(x, bridge=signal)
                decs.append(x)
                sam, output = self.stage_0_supervised_attention_module_2(x, shortcuts[i])
                outputs.append(output)
                sam_features.append(sam)
            elif i == 1:
                # get multi-scale skip signals from cross-gating block
                signal2 = self.UpSampleRatio_12(skip_features[0])
                signal1 = self.UpSampleRatio_13(skip_features[1])
                signal0 = self.UpSampleRatio_14(skip_features[2])
                signal = torch.cat([signal2, signal1, signal0], dim=1)
                # Decoder block
                x = self.stage_0_decoder_block_1(x, bridge=signal)
                decs.append(x)
                sam, output = self.stage_0_supervised_attention_module_1(x, shortcuts[i])
                outputs.append(output)
                sam_features.append(sam)
            elif i == 0:
                # get multi-scale skip signals from cross-gating block
                signal2 = self.UpSampleRatio_15(skip_features[0])
                signal1 = self.UpSampleRatio_16(skip_features[1])
                signal0 = self.UpSampleRatio_17(skip_features[2])
                signal = torch.cat([signal2, signal1, signal0], dim=1)
                # Decoder block
                x = self.stage_0_decoder_block_0(x, bridge=signal)
                decs.append(x)
                sam, output = self.stage_0_supervised_attention_module_0(x, shortcuts[i])
                outputs.append(output)
                sam_features.append(sam)

        # Cache encoder and decoder features for later-stage's usage
        encs_prev = encs[::-1]
        decs_prev = decs
        # Store outputs
        outputs_all.append(outputs)

        ########## STAGE 1 ##########
        x_scales = []
        for i in range(self.num_supervision_scales):
            if i == 0:
                x_scale = self.stage_1_input_conv_0(shortcuts[i])
                # If later stages, fuse input features with SAM features from prev stage
                x_scale, _ = self.stage_1_input_fuse_sam_0(x_scale, sam_features.pop())
                x_scales.append(x_scale)
            elif i == 1:
                x_scale = self.stage_1_input_conv_1(shortcuts[i])
                # If later stages, fuse input features with SAM features from prev stage
                x_scale, _ = self.stage_1_input_fuse_sam_1(x_scale, sam_features.pop())
                x_scales.append(x_scale)
            else:
                x_scale = self.stage_1_input_conv_2(shortcuts[i])
                # If later stages, fuse input features with SAM features from prev stage
                x_scale, _ = self.stage_1_input_fuse_sam_2(x_scale, sam_features.pop())
                x_scales.append(x_scale)
        # start encoder blocks
        encs = []
        x = x_scales[0]  # First full-scale input feature
        # use larger blocksize at high-res stages, vice versa.
        for i in range(self.depth):
            x_scale = x_scales[i] if i < self.num_supervision_scales else None
            enc_prev = encs_prev.pop()
            dec_prev = decs_prev.pop()
            if i == 0:
                x, bridge = self.stage_1_encoder_block_0(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
            elif i == 1:
                x, bridge = self.stage_1_encoder_block_1(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
            else:
                x, bridge = self.stage_1_encoder_block_2(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
            encs.append(bridge)
        # Global MLP bottleneck blocks
        x = self.stage_1_global_block_0(x)
        x = self.stage_1_global_block_1(x)

        # cache global feature for cross-gating
        global_feature = x

        # start cross gating. Use multi-scale feature fusion
        skip_features = []
        for i in reversed(range(self.depth)):  # 2, 1, 0
            if i == 2:
                # get multi-scale skip signals from cross-gating block
                signal0 = self.UpSampleRatio_18(encs[0])
                signal1 = self.UpSampleRatio_19(encs[1])
                signal2 = self.UpSampleRatio_20(encs[2])
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                # Use cross-gating to cross modulate features
                skips, global_feature = self.stage_1_cross_gating_block_2(signal, global_feature)
                skip_features.append(skips)
            elif i == 1:
                # get multi-scale skip signals from cross-gating block
                signal0 = self.UpSampleRatio_21(encs[0])
                signal1 = self.UpSampleRatio_22(encs[1])
                signal2 = self.UpSampleRatio_23(encs[2])
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                # Use cross-gating to cross modulate features
                skips, global_feature = self.stage_1_cross_gating_block_1(signal, global_feature)
                skip_features.append(skips)
            elif i == 0:
                # get multi-scale skip signals from cross-gating block
                signal0 = self.UpSampleRatio_24(encs[0])
                signal1 = self.UpSampleRatio_25(encs[1])
                signal2 = self.UpSampleRatio_26(encs[2])
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                # Use cross-gating to cross modulate features
                skips, global_feature = self.stage_1_cross_gating_block_0(signal, global_feature)
                skip_features.append(skips)

        # start decoder. Multi-scale feature fusion of cross-gated features
        outputs, decs, sam_features = [], [], []
        for i in reversed(range(self.depth)):
            if i == 2:
                # get multi-scale skip signals from cross-gating block
                signal2 = self.UpSampleRatio_27(skip_features[0])
                signal1 = self.UpSampleRatio_28(skip_features[1])
                signal0 = self.UpSampleRatio_29(skip_features[2])
                signal = torch.cat([signal2, signal1, signal0], dim=1)
                # Decoder block
                x = self.stage_1_decoder_block_2(x, bridge=signal)
                decs.append(x)
                # output conv, if not final stage, use supervised-attention-block.
                # not last stage, apply SAM
                sam, output = self.stage_1_supervised_attention_module_2(x, shortcuts[i])
                outputs.append(output)
                sam_features.append(sam)
            elif i == 1:
                # get multi-scale skip signals from cross-gating block
                signal2 = self.UpSampleRatio_30(skip_features[0])
                signal1 = self.UpSampleRatio_31(skip_features[1])
                signal0 = self.UpSampleRatio_32(skip_features[2])
                signal = torch.cat([signal2, signal1, signal0], dim=1)
                # Decoder block
                x = self.stage_1_decoder_block_1(x, bridge=signal)
                decs.append(x)
                # output conv, if not final stage, use supervised-attention-block.
                # not last stage, apply SAM
                sam, output = self.stage_1_supervised_attention_module_1(x, shortcuts[i])
                outputs.append(output)
                sam_features.append(sam)
            elif i == 0:
                # get multi-scale skip signals from cross-gating block
                signal2 = self.UpSampleRatio_33(skip_features[0])
                signal1 = self.UpSampleRatio_34(skip_features[1])
                signal0 = self.UpSampleRatio_35(skip_features[2])
                signal = torch.cat([signal2, signal1, signal0], dim=1)
                # Decoder block
                x = self.stage_1_decoder_block_0(x, bridge=signal)
                decs.append(x)
                # output conv, if not final stage, use supervised-attention-block.
                # not last stage, apply SAM
                sam, output = self.stage_1_supervised_attention_module_0(x, shortcuts[i])
                outputs.append(output)
                sam_features.append(sam)

        # Cache encoder and decoder features for later-stage's usage
        encs_prev = encs[::-1]
        decs_prev = decs
        # Store outputs
        outputs_all.append(outputs)

        ########## STAGE 2 ##########
        x_scales = []
        for i in range(self.num_supervision_scales):
            if i == 0:
                x_scale = self.stage_2_input_conv_0(shortcuts[i])
                # If later stages, fuse input features with SAM features from prev stage
                x_scale, _ = self.stage_2_input_fuse_sam_0(x_scale, sam_features.pop())
                x_scales.append(x_scale)
            elif i == 1:
                x_scale = self.stage_2_input_conv_1(shortcuts[i])
                # If later stages, fuse input features with SAM features from prev stage
                x_scale, _ = self.stage_2_input_fuse_sam_1(x_scale, sam_features.pop())
                x_scales.append(x_scale)
            else:
                x_scale = self.stage_2_input_conv_2(shortcuts[i])
                # If later stages, fuse input features with SAM features from prev stage
                x_scale, _ = self.stage_2_input_fuse_sam_2(x_scale, sam_features.pop())
                x_scales.append(x_scale)
        # start encoder blocks
        encs = []
        x = x_scales[0]  # First full-scale input feature
        # use larger blocksize at high-res stages, vice versa.
        for i in range(self.depth):
            x_scale = x_scales[i] if i < self.num_supervision_scales else None
            enc_prev = encs_prev.pop()
            dec_prev = decs_prev.pop()
            if i == 0:
                x, bridge = self.stage_2_encoder_block_0(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
            elif i == 1:
                x, bridge = self.stage_2_encoder_block_1(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
            else:
                x, bridge = self.stage_2_encoder_block_2(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
            encs.append(bridge)
        # Global MLP bottleneck blocks
        x = self.stage_2_global_block_0(x)
        x = self.stage_2_global_block_1(x)

        # cache global feature for cross-gating
        global_feature = x

        # start cross gating. Use multi-scale feature fusion
        skip_features = []
        for i in reversed(range(self.depth)):  # 2, 1, 0
            if i == 2:
                # get multi-scale skip signals from cross-gating block
                signal0 = self.UpSampleRatio_36(encs[0])
                signal1 = self.UpSampleRatio_37(encs[1])
                signal2 = self.UpSampleRatio_38(encs[2])
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                # Use cross-gating to cross modulate features
                skips, global_feature = self.stage_2_cross_gating_block_2(signal, global_feature)
                skip_features.append(skips)
            elif i == 1:
                # get multi-scale skip signals from cross-gating block
                signal0 = self.UpSampleRatio_39(encs[0])
                signal1 = self.UpSampleRatio_40(encs[1])
                signal2 = self.UpSampleRatio_41(encs[2])
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                # Use cross-gating to cross modulate features
                skips, global_feature = self.stage_2_cross_gating_block_1(signal, global_feature)
                skip_features.append(skips)
            elif i == 0:
                # get multi-scale skip signals from cross-gating block
                signal0 = self.UpSampleRatio_42(encs[0])
                signal1 = self.UpSampleRatio_43(encs[1])
                signal2 = self.UpSampleRatio_44(encs[2])
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                # Use cross-gating to cross modulate features
                skips, global_feature = self.stage_2_cross_gating_block_0(signal, global_feature)
                skip_features.append(skips)

        # start decoder. Multi-scale feature fusion of cross-gated features
        outputs = []
        for i in reversed(range(self.depth)):
            if i == 2:
                # get multi-scale skip signals from cross-gating block
                signal2 = self.UpSampleRatio_45(skip_features[0])
                signal1 = self.UpSampleRatio_46(skip_features[1])
                signal0 = self.UpSampleRatio_47(skip_features[2])
                signal = torch.cat([signal2, signal1, signal0], dim=1)
                # Decoder block
                x = self.stage_2_decoder_block_2(x, bridge=signal)
                decs.append(x)
                # output conv, if not final stage, use supervised-attention-block.
                # Last stage, apply output convolutions
                output = self.stage_2_output_conv_2(x)
                output = output + shortcuts[i]
                outputs.append(output)
            elif i == 1:
                # get multi-scale skip signals from cross-gating block
                signal2 = self.UpSampleRatio_48(skip_features[0])
                signal1 = self.UpSampleRatio_49(skip_features[1])
                signal0 = self.UpSampleRatio_50(skip_features[2])
                signal = torch.cat([signal2, signal1, signal0], dim=1)
                # Decoder block
                x = self.stage_2_decoder_block_1(x, bridge=signal)
                decs.append(x)
                # output conv, if not final stage, use supervised-attention-block.
                # Last stage, apply output convolutions
                output = self.stage_2_output_conv_1(x)
                output = output + shortcuts[i]
                outputs.append(output)
            elif i == 0:
                # get multi-scale skip signals from cross-gating block
                signal2 = self.UpSampleRatio_51(skip_features[0])
                signal1 = self.UpSampleRatio_52(skip_features[1])
                signal0 = self.UpSampleRatio_53(skip_features[2])
                signal = torch.cat([signal2, signal1, signal0], dim=1)
                # Decoder block
                x = self.stage_2_decoder_block_0(x, bridge=signal)
                decs.append(x)
                # output conv, if not final stage, use supervised-attention-block.
                # Last stage, apply output convolutions
                output = self.stage_2_output_conv_0(x)
                output = output + shortcuts[i]
                outputs.append(output)

        # Store outputs
        outputs_all.append(outputs)

        return outputs_all

# ------------------------------------------------------------------------------
# DeepLabV3+ decoder.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .aspp import ASPP
from .conv_module import stacked_conv


__all__ = ["DeepLabV3PlusDecoder"]


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project,
                 decoder_channels, atrous_rates, num_classes):
        super(DeepLabV3PlusDecoder, self).__init__()
        self.aspp = ASPP(in_channels, out_channels=decoder_channels, atrous_rates=atrous_rates)
        self.feature_key = feature_key
        self.low_level_key = low_level_key
        # Transform low-level feature
        # low_level_channels_project = 48
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_channels_project, 1, bias=False),
            nn.BatchNorm2d(low_level_channels_project),
            nn.ReLU()
        )
        # Fuse
        self.fuse = stacked_conv(
            decoder_channels + low_level_channels_project,
            decoder_channels,
            kernel_size=3,
            padding=1,
            num_stack=2,
            conv_type='depthwise_separable_conv'
        )
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features):
        pred = OrderedDict()
        l = features[self.low_level_key]
        x = features[self.feature_key]
        x = self.aspp(x)
        # low-level feature
        l = self.project(l)
        x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, l), dim=1)
        x = self.fuse(x)
        x = self.classifier(x)
        pred['semantic'] = x
        return pred

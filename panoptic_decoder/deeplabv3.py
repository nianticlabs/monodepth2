# ------------------------------------------------------------------------------
# DeepLabV3 decoder.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict

from torch import nn

from .aspp import ASPP


__all__ = ["DeepLabV3Decoder"]


class DeepLabV3Decoder(nn.Module):
    def __init__(self, in_channels, feature_key, decoder_channels, atrous_rates, num_classes):
        super(DeepLabV3Decoder, self).__init__()
        self.aspp = ASPP(in_channels, out_channels=decoder_channels, atrous_rates=atrous_rates)
        self.feature_key = feature_key
        self.classifier = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(),
            nn.Conv2d(decoder_channels, num_classes, 1)
        )

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features):
        pred = OrderedDict()
        res5 = features[self.feature_key]
        x = self.aspp(res5)
        x = self.classifier(x)
        pred['semantic'] = x
        return pred

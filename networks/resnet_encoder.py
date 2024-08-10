# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, ResNet18_Weights, ResNet50_Weights

class ResNetMultiImageInput(ResNet):
    """Constructs a ResNet model with a varying number of input images."""
    def __init__(self, block, layers, num_input_images=1, **kwargs):
        super(ResNetMultiImageInput, self).__init__(block, layers, **kwargs)
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, progress=True):
    """Constructs a ResNet model with varying number of input images.
    Args:
        num_layers (int): Number of ResNet layers. Must be 18 or 50.
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        num_input_images (int): Number of frames stacked as input.
        progress (bool): If True, displays a progress bar of the download.
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer ResNet"
    
    layers = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block = {18: BasicBlock, 50: Bottleneck}[num_layers]
    
    # Choose the correct weights based on the num_layers
    if num_layers == 18:
        weights_enum = ResNet18_Weights if pretrained else None
        weights = ResNet18_Weights.DEFAULT if pretrained else None
    elif num_layers == 50:
        weights_enum = ResNet50_Weights if pretrained else None
        weights = ResNet50_Weights.DEFAULT if pretrained else None

    model = ResNetMultiImageInput(block, layers, num_input_images=num_input_images)

    if pretrained:
        state_dict = weights.get_state_dict(progress=progress, check_hash=True)
        
        # Modify the conv1 weight to accommodate multiple input images
        conv1_weight = state_dict['conv1.weight']
        if num_input_images > 1:
            # Repeat the conv1 weights for the additional image channels
            conv1_weight = conv1_weight.repeat(1, num_input_images, 1, 1) / num_input_images
        
        state_dict['conv1.weight'] = conv1_weight
        
        # Load the modified state dict into the model
        model.load_state_dict(state_dict, strict=False)

    return model

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

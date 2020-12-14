# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
<<<<<<< Updated upstream
=======
from networks.ConvRNN import CGRU_cell
from networks.convlstm import ConvLSTM
>>>>>>> Stashed changes


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
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

<<<<<<< Updated upstream
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
=======
        # For Resnet18
        #self.convlstm4 = CGRU_cell(shape=(6,20), input_channels=512, filter_size=3, num_features=512)
        self.convlstm1 = ConvLSTM(64,64, (3,3), 1, True, True,  return_all_layers=False).to('cuda')
        self.convlstm2 = ConvLSTM(64,64, (3,3), 1, True, True,  return_all_layers=False).to('cuda')
        self.convlstm3 = ConvLSTM(128,128, (3,3), 1, True, True,  return_all_layers=False).to('cuda')
        self.convlstm4 = ConvLSTM(256,256, (3,3), 1, True, True,  return_all_layers=False).to('cuda')
        self.convlstm5 = ConvLSTM(512,512, (3,3), 1, True, True,  return_all_layers=False).to('cuda')
        self.hidden_state1=torch.autograd.Variable(torch.zeros(1, 64, 96, 320), requires_grad=True).cuda()
        self.hidden_state2=torch.autograd.Variable(torch.zeros(1, 64, 48, 160), requires_grad=True).cuda()
        self.hidden_state3=torch.autograd.Variable(torch.zeros(1, 128, 24, 80), requires_grad=True).cuda()
        self.hidden_state4=torch.autograd.Variable(torch.zeros(1, 256, 12, 40), requires_grad=True).cuda()
        self.hidden_state5=torch.autograd.Variable(torch.zeros(1, 512, 6, 20), requires_grad=True).cuda()

    def forward(self, input_image,hidden_state=None):
            
        last_state_list=[]
        self.features = []
        x = (input_image - 0.45) / 0.225
        batch_size,seq_number, input_channel, height, width = x.shape


        last_state1=[[self.hidden_state1.repeat(x.size(0),1,1,1),torch.zeros(1, 64, 96, 320,device='cuda')]]
        last_state2=[[self.hidden_state2.repeat(x.size(0),1,1,1),torch.zeros(1, 64, 48, 160,device='cuda')]]
        last_state3=[[self.hidden_state3.repeat(x.size(0),1,1,1),torch.zeros(1, 128, 24, 80,device='cuda')]]
        last_state4=[[self.hidden_state4.repeat(x.size(0),1,1,1),torch.zeros(1, 256, 12, 40,device='cuda')]]
        last_state5=[[self.hidden_state5.repeat(x.size(0),1,1,1),torch.zeros(1, 512, 6, 20,device='cuda')]]
   
        # Multi Resolution LSTM Encoder
        # LSTM 1
        x = torch.reshape(x, (-1, input_channel, height, width))
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = torch.reshape(x, (batch_size,seq_number, x.size(1), x.size(2), x.size(3)))
        features,last_state = self.convlstm1(x,last_state1)
        self.features.append(features[0].permute(1, 0, 2, 3, 4)[-1])
        last_state_list.append(last_state)

        # LSTM 2
        x = torch.reshape(features[0], (-1, x.size(2), x.size(3), x.size(4)))
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = torch.reshape(x, (batch_size,seq_number, x.size(1), x.size(2), x.size(3)))
        features,last_state = self.convlstm2(x,last_state2)
        self.features.append(features[0].permute(1, 0, 2, 3, 4)[-1])
        last_state_list.append(last_state)
        
        # LSTM 3
        x = torch.reshape(features[0], (-1, x.size(2), x.size(3), x.size(4)))
        x = self.encoder.layer2(x)
        x = torch.reshape(x, (batch_size,seq_number, x.size(1), x.size(2), x.size(3)))
        features,last_state = self.convlstm3(x,last_state3)
        self.features.append(features[0].permute(1, 0, 2, 3, 4)[-1])
        last_state_list.append(last_state)
        
        # LSTM 4
        x = torch.reshape(features[0], (-1, x.size(2), x.size(3), x.size(4)))
        x = self.encoder.layer3(x)
        x = torch.reshape(x, (batch_size,seq_number, x.size(1), x.size(2), x.size(3)))
        features,last_state = self.convlstm4(x,last_state4)
        self.features.append(features[0].permute(1, 0, 2, 3, 4)[-1])
        last_state_list.append(last_state)

        # LSTM 5
        x = torch.reshape(features[0], (-1, x.size(2), x.size(3), x.size(4)))
        x = self.encoder.layer4(x)
        x = torch.reshape(x, (batch_size,seq_number, x.size(1), x.size(2), x.size(3)))
        #if hidden_state is None:
           # _, self.last_states = self.convlstm(x)
        #else:
        features, last_state = self.convlstm5(x,last_state5)
        self.features.append(features[0].permute(1, 0, 2, 3, 4)[-1])
        last_state_list.append(last_state)

        #print(x.shape)
        #print(seq_number)
        #output, hidden = self.convlstm4(x, hidden_state=self.hidden_state, seq_len=seq_number) 
        return self.features,last_state_list
>>>>>>> Stashed changes

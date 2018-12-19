"""
appearance_model.py

Segmentation network that uses only appearance information.

Based on:
    - https://github.com/kmaninis/OSVOS-PyTorch/blob/master/networks/vgg_osvos.py
      Written by Kevis-Kokitsi Maninis.
"""

import copy
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.modules as modules
import torchvision

from .helper import make_layers, get_conv_layers, center_crop, interp_surgery
from torch.nn import functional as F

class AppearanceModel(nn.Module):
    def __init__(self, pretrained=True):
        super(AppearanceModel, self).__init__()
        print("Setting up Appearance model...", end="")
        sys.stdout.flush()

        self.n_blocks = 5
        layers_list = [[64, 64],
                    ["M", 128, 128],
                    ["M", 256, 256, 256],
                    ["M", 512, 512, 512],
                    ["M", 512, 512, 512]]
        in_channels = [3, 64, 128, 256, 512]

        blocks = modules.ModuleList()
        sides = modules.ModuleList()
        upsamples = modules.ModuleList()

        for i in range(self.n_blocks):
            blocks.append(make_layers(layers_list[i], in_channels[i]))

            if i > 0:
                sides.append(
                    nn.Conv2d(layers_list[i][-1], 16, kernel_size=3, padding=1))
                upsamples.append(
                    nn.ConvTranspose2d(16, 16, kernel_size=2**(i+1), 
                                       stride=2**i, bias=False))

        self.blocks = blocks
        self.sides = sides
        self.upsamples = upsamples

        self.test = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, bias=False)

        self.fuse = nn.Conv2d(64, 1, kernel_size=1)

        if pretrained:
            print("Initializing weights...", end="")
            sys.stdout.flush()
            self._initialize_weights(pretrained)

        print("Done!")

    def forward(self, x):
        orig_h, orig_w = x.size()[2], x.size()[3]
        x = self.blocks[0](x)

        sides = []
        for i in range(1, self.n_blocks):
            x = self.blocks[i](x)
            side = self.sides[i-1](x)
            side_upsampled = center_crop(self.upsamples[i-1](side), 
                                         orig_h, orig_w)
            sides.append(side_upsampled)

        out = torch.cat(sides, dim=1)
        out = self.fuse(out)

        return out


    def _initialize_weights(self, pretrained):

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.zero_()
                m.weight.data = interp_surgery(m)

        print("Loading weights from PyTorch VGG...", end="")
        sys.stdout.flush()
        vgg_model = torchvision.models.vgg.vgg16(pretrained=True)

        vgg_conv_layers = get_conv_layers(vgg_model)

        k = 0

        for i in range(self.n_blocks):
            for j in range(len(self.blocks[i])):
                if isinstance(self.blocks[i][j], nn.Conv2d):
                    self.blocks[i][j].weight = copy.deepcopy(
                        vgg_conv_layers[k].weight)
                    self.blocks[i][j].bias = copy.deepcopy(
                        vgg_conv_layers[k].bias)
                    k += 1

        assert k == len(vgg_conv_layers)


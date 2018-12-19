"""
appearance_flow_model.py

Segmentation network that uses both appearance & flow information.

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

class AppearanceFlowModel(nn.Module):
    def __init__(self, pretrained=True, mask=False):
        super(AppearanceFlowModel, self).__init__()
        print("Setting up Appearance+Flow model...", end="")
        sys.stdout.flush()

        self.mask = mask
        self.n_blocks = 5
        layers_list = [[64, 64],
                    ["M", 128, 128],
                    ["M", 256, 256, 256],
                    ["M", 512, 512, 512],
                    ["M", 512, 512, 512]]
        app_in_channels = [3, 64, 128, 256, 512]
        flow_in_channels = [3, 64, 128, 256, 512]
        if mask:
            app_in_channels[0] = 4
            flow_in_channels[0] = 4

        # Make 2 separate streams.
        app_blocks = modules.ModuleList()
        app_sides = modules.ModuleList()

        flow_blocks = modules.ModuleList()
        flow_sides = modules.ModuleList()

        upsamples = modules.ModuleList()

        # Make appearance stream.
        for i in range(self.n_blocks):
            app_blocks.append(make_layers(layers_list[i], app_in_channels[i]))

            if i > 0:
                app_sides.append(
                    nn.Conv2d(layers_list[i][-1], 16, kernel_size=3, padding=1))
                upsamples.append(
                    nn.ConvTranspose2d(16, 16, kernel_size=2**(i+1), 
                                       stride=2**i, bias=False))
        self.app_blocks = app_blocks
        self.app_sides = app_sides

        # Make flow stream.
        for i in range(self.n_blocks):
            flow_blocks.append(make_layers(layers_list[i], flow_in_channels[i]))

            if i > 0:
                flow_sides.append(
                    nn.Conv2d(layers_list[i][-1], 16, kernel_size=3, padding=1))

        self.flow_blocks = flow_blocks
        self.flow_sides = flow_sides

        self.upsamples = upsamples
        self.fuse = nn.Conv2d(128, 1, kernel_size=1)

        if pretrained:
            print("Initializing weights...", end="")
            sys.stdout.flush()
            self._initialize_weights(pretrained)

        # Make upsample layers not trainable.
        for param in self.upsamples.parameters():
            param.requires_grad = False

        print("Done!")

    def forward(self, img, flow, mask_proposal=None):
        orig_h, orig_w = img.size()[2], img.size()[3]

        if mask_proposal is not None:
            img = torch.cat([img, mask_proposal], 1)
            flow = torch.cat([flow, mask_proposal], 1)

        I = self.app_blocks[0](img)
        F = self.flow_blocks[0](flow)

        sides = []
        for i in range(1, self.n_blocks):
            I = self.app_blocks[i](I)
            side = self.app_sides[i-1](I)
            side_upsampled = center_crop(self.upsamples[i-1](side), 
                                         orig_h, orig_w)
            sides.append(side_upsampled)

        for i in range(1, self.n_blocks):
            F = self.flow_blocks[i](F)
            side = self.flow_sides[i-1](F)
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
            for j in range(len(self.app_blocks[i])):
                if self.mask and i == 0 and j == 0:
                    k += 1
                    continue
                if isinstance(self.app_blocks[i][j], nn.Conv2d):
                    self.app_blocks[i][j].weight = copy.deepcopy(
                        vgg_conv_layers[k].weight)
                    self.app_blocks[i][j].bias = copy.deepcopy(
                        vgg_conv_layers[k].bias)
                    self.flow_blocks[i][j].weight = copy.deepcopy(
                        vgg_conv_layers[k].weight)
                    self.flow_blocks[i][j].bias = copy.deepcopy(
                        vgg_conv_layers[k].bias)
                    k += 1

        assert k == len(vgg_conv_layers)

        self.app_blocks[0][0].weight.data[:,:3,:,:] = copy.deepcopy(
            vgg_conv_layers[0].weight.data)
        self.app_blocks[0][0].bias = copy.deepcopy(vgg_conv_layers[0].bias)
        self.flow_blocks[0][0].weight.data[:,:3,:,:] = copy.deepcopy(
            vgg_conv_layers[0].weight.data)
        self.flow_blocks[0][0].bias = copy.deepcopy(vgg_conv_layers[0].bias)

    def set_finetune_flow(self, true_or_false):
        for param in self.flow_blocks:
            param.requires_grad = true_or_false
        for param in self.flow_sides:
            param.requires_grad = true_or_false

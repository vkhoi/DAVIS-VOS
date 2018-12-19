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
from torch.autograd import Variable

from .helper import make_layers, get_conv_layers, center_crop, interp_surgery
from torch.nn import functional as F

from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.bbox_transform import bbox_transform_inv, bbox_transform_batch
from model.utils.net_utils import _smooth_l1_loss


class AppearanceFlowLocalizedModel(nn.Module):
    def __init__(self, pretrained=True):
        super(AppearanceFlowLocalizedModel, self).__init__()
        print("Setting up Appearance+Flow+Localized model...")
        sys.stdout.flush()
        self.use_cuda = False
        self.n_blocks = 5
        layers_list = [[64, 64],
                    ["M", 128, 128],
                    ["M", 256, 256, 256],
                    ["M", 512, 512, 512],
                    ["M", 512, 512, 512]]
        app_in_channels = [3, 64, 128, 256, 512]
        flow_in_channels = [3, 64, 128, 256, 512]

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
            print("Initializing weights...")
            sys.stdout.flush()
            self._initialize_weights(pretrained)

        # Make upsample layers not trainable.
        for param in self.upsamples.parameters():
            param.requires_grad = False

        # Setup localization stream.
        print("Setting up localization stream...", end="")
        sys.stdout.flush()
        vgg = torchvision.models.vgg.vgg16(pretrained=True)
        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        self.RCNN_top = vgg.classifier
        POOLING_MODE = 'crop'
        POOLING_SIZE = 7
        self.RCNN_bbox_pred = nn.Linear(4096, 4)
        self.RCNN_roi_pool = _RoIPooling(POOLING_SIZE, POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(POOLING_SIZE, POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_crop = _RoICrop()
        print("Done!")

    def forward(self, img, flow, mask_prev=None, boxes=None):
        orig_h, orig_w = img.size()[2], img.size()[3]

        if mask_prev is not None:
            # img = torch.cat([img, mask_proposal], 1)
            # flow = torch.cat([flow, mask_proposal], 1)
            flow[0,2,:,:].data = mask_prev[0,:,:].data.clone()

        #calculate boxes from img

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

        ''' Object localization Network 
        Input:  conv5_3 (= I, after appearance branch, N*64*w*h )
                boxes, Bounding Box proposal (N*4, )
        '''
        if boxes is not None:
            base_feat = I
            rois = Variable(boxes)
            rois = rois.view(-1,5)
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
            pooled_feat = self._head_to_tail(pooled_feat)

            bbox_pred = self.RCNN_bbox_pred(pooled_feat)

            # if gt_box is not None:
            #     rois_target = bbox_transform_batch(boxes[:,:,1:5], gt_box[:,:,:4])
            #     rois_inside_ws = rois_target.new(rois_target.size()).zero_()
            #     rois_inside_ws[:, :, :] = 0.5
            #     rois_outside_ws = (rois_inside_ws > 0).float()
            #     rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            #     rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            #     rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            #     RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            # else:
            #     RCNN_loss_bbox = 0

            box_deltas = bbox_pred.data
            box_deltas = box_deltas.view(boxes.size(0), boxes.size(1), -1)
            refined_bounding_box = bbox_transform_inv(boxes, box_deltas, 1)
        else:
            bbox_pred = None
            refined_bounding_box = None

        # return out, RCNN_loss_bbox, refined_bounding_box
        return out, bbox_pred, refined_bounding_box

    def _initialize_weights(self, pretrained):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.zero_()
                m.weight.data = interp_surgery(m)

        print("Loading weights from PyTorch VGG...")
        sys.stdout.flush()
        vgg_model = torchvision.models.vgg.vgg16(pretrained=True)

        vgg_conv_layers = get_conv_layers(vgg_model)

        k = 0

        for i in range(self.n_blocks):
            for j in range(len(self.app_blocks[i])):
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

    def set_finetune_flow(self, true_or_false):
        for param in self.flow_blocks:
            param.requires_grad = true_or_false
        for param in self.flow_sides:
            param.requires_grad = true_or_false
    def _head_to_tail(self, pool5):
    
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7
    def use_cuda():
        self.use_cuda = True



#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Tencent, Inc. All Rights Reserved
#
"""
File: unet3d.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2019-5-25 22:15:25
"""

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from .spacingBlock import SpacingBlock, SpacingBlock1, SpacingBlock2, SpacingBlock3


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, weight=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv.bias.data.zero_()
        # self.bn = nn.BatchNorm3d(out_channels)
        self.bn = nn.GroupNorm(num_groups=int(out_channels//4), num_channels=out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        return self.ReLU(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        nn.init.kaiming_normal_(self.deconv.weight)
        if self.deconv.bias is not None:
            self.deconv.bias.data.zero_()
        # self.bn = nn.BatchNorm3d(out_channels)
        self.bn = nn.GroupNorm(num_groups=int(out_channels//4), num_channels=out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        return self.ReLU(self.bn(self.deconv(x)))


class SpacingUnet(nn.Module):
    def __init__(self, num_classes=2, spacingblock=SpacingBlock, **kwargs):
        super(SpacingUnet, self).__init__()
        self.conv1 = ConvBlock(1, 32, None)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.conv2 = ConvBlock(32, 64, None)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.conv3 = ConvBlock(64, 128, None)
        self.conv3_1 = ConvBlock(128, 128, None)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.conv4 = ConvBlock(128, 128, None)
        self.conv4_1 = spacingblock(in_channels=128, sub_sample=True)
        self.conv4_2 = ConvBlock(128, 128, None)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.conv5 = ConvBlock(128, 128, None)
        self.conv5_1 = ConvBlock(128, 128, None)
        self.deconv0_1 = DeconvBlock(128, 128)
        self.deconv0_2 = ConvBlock(128+128, 128, None)
        self.deconv1_1 = DeconvBlock(128, 64)
        self.deconv1_2 = SpacingBlock(in_channels=64+128)
        self.deconv1_3 = ConvBlock(64+128, 64, None)
        self.deconv2_1 = DeconvBlock(64, 32)
        self.deconv2_2 = ConvBlock(32+64, 32, None)
        self.deconv3_1 = DeconvBlock(32, 16)
        self.deconv3_2 = ConvBlock(16+32, 16, None)
        self.conv8 = nn.Conv3d(16, num_classes, kernel_size=(3, 3, 1), padding=(1, 1, 0))

    def forward(self, x, spacings):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        conv3_1 = self.conv3_1(conv3)
        pool3 = self.pool3(conv3_1)
        conv4 = self.conv4(pool3)
        conv4_1 = self.conv4_1(conv4, spacings)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.pool4(conv4_2)
        conv5 = self.conv5(pool4)
        conv5_1 = self.conv5_1(conv5)
        deconv0_1 = self.deconv0_1(conv5_1)
        deconv0_2 = self.deconv0_2(torch.cat([deconv0_1, conv4_2], 1))
        deconv1_1 = self.deconv1_1(deconv0_2)
        deconv1_2 = self.deconv1_2(torch.cat([deconv1_1, conv3_1], 1), spacings)
        deconv1_3 = self.deconv1_3(deconv1_2)
        deconv2_1 = self.deconv2_1(deconv1_3)
        deconv2_2 = self.deconv2_2(torch.cat([deconv2_1, conv2], 1))
        deconv3_1 = self.deconv3_1(deconv2_2)
        deconv3_2 = self.deconv3_2(torch.cat([deconv3_1, conv1], 1))
        output = self.conv8(deconv3_2)

        # make channels the last axis
        output = output.permute(0, 2, 3, 4, 1).contiguous()
        return output


class SpacingUnet0(SpacingUnet):
    def __init__(self, num_classes=1, **kwargs):
        super(SpacingUnet0, self).__init__(num_classes=num_classes, spacingblock=SpacingBlock)


class SpacingUnet1(SpacingUnet):
    def __init__(self, num_classes=1, **kwargs):
        super(SpacingUnet1, self).__init__(num_classes=num_classes, spacingblock=SpacingBlock1)


class SpacingUnet2(SpacingUnet):
    def __init__(self, num_classes=1, **kwargs):
        super(SpacingUnet2, self).__init__(num_classes=num_classes, spacingblock=SpacingBlock2)


class SpacingUnet3(SpacingUnet):
    def __init__(self, num_classes=1, **kwargs):
        super(SpacingUnet3, self).__init__(num_classes=num_classes, spacingblock=SpacingBlock3)


if __name__ == '__main__':
    import time
    t = time.time()
    net = SpacingUnet3()
    x = torch.ones((2, 1, 64, 64, 16))
    spacing = torch.from_numpy(np.asarray([[0.8, 0.8, 0.8], [2.6, 1.2, 2.5]]))
    output = net(x, spacing)
    print(output.size())
    print(time.time()-t)

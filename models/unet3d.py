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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, weight=None, kernel_size=(3, 3, 1), padding=(1, 1, 0)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        if weight is not None:
            print(weight[0].shape)
            self.conv.weight = nn.Parameter(torch.from_numpy(np.transpose(weight[0], (4, 3, 0, 1, 2))))
            self.conv.bias = nn.Parameter(torch.from_numpy(weight[1]))
        else:
            nn.init.kaiming_normal_(self.conv.weight)
            if self.conv.bias is not None:
                self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm3d(out_channels)
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
        self.bn = nn.BatchNorm3d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        return self.ReLU(self.bn(self.deconv(x)))


class Unet3D(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super(Unet3D, self).__init__()
        self.conv1 = ConvBlock(1, 32, None)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.conv2 = ConvBlock(32, 64, None)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.conv3_1 = ConvBlock(64, 128, None)
        self.conv3_2 = ConvBlock(128, 128, None)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.conv4_1 = ConvBlock(128, 128, None)
        self.conv4_2 = ConvBlock(128, 128, None)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.conv5_1 = ConvBlock(128, 128, None)
        self.conv5_2 = ConvBlock(128, 128, None)
        self.deconv0_1 = DeconvBlock(128, 128)
        self.deconv0_2 = ConvBlock(128+128, 128, None)
        self.deconv1_1 = DeconvBlock(128, 64)
        self.deconv1_2 = ConvBlock(64+128, 64, None)
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
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.pool3(conv3_2)
        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.pool4(conv4_2)
        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        deconv0_1 = self.deconv0_1(conv5_2)
        deconv0_2 = self.deconv0_2(torch.cat([deconv0_1, conv4_2], 1))
        deconv1_1 = self.deconv1_1(deconv0_2)
        deconv1_2 = self.deconv1_2(torch.cat([deconv1_1, conv3_2], 1))
        deconv2_1 = self.deconv2_1(deconv1_2)
        deconv2_2 = self.deconv2_2(torch.cat([deconv2_1, conv2], 1))
        deconv3_1 = self.deconv3_1(deconv2_2)
        deconv3_2 = self.deconv3_2(torch.cat([deconv3_1, conv1], 1))
        output = self.conv8(deconv3_2)

        # make channels the last axis
        output = output.permute(0, 2, 3, 4, 1).contiguous()
        return output


if __name__ == '__main__':
    import time

    t = time.time()
    net = Unet3D()
    x = torch.ones((2, 1, 64, 64, 16))
    spacing = torch.from_numpy(np.asarray([[0.8, 0.8, 0.8], [2.6, 1.2, 2.5]]))
    output = net(x, spacing)
    print(output.size())
    print(time.time() - t)

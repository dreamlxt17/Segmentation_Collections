#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Tencent, Inc. All Rights Reserved
#
"""
File: pspnet.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2018/6/19 17:41
"""

import torch
import torch.nn.functional as F
from torch import nn
from .resnet3d import resnet50, resnet34
from .unet3d import ConvBlock, DeconvBlock
from tumor_config import Config
config = Config()


class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool3d(s),  # output zize: sxs
                nn.Conv3d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm3d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)))

        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='trilinear'))
        out = torch.cat(out, 1)
        return out


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class PSPNet(nn.Module):
    def __init__(self, pretrained='', num_classes=1, **kwargs):
        super(PSPNet, self).__init__()
        resnet = resnet50()
        if pretrained:
            resnet.load_state_dict(torch.load(pretrained))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = 2, 2, 1
            elif 'downsample.0' in n:
                m.stride = 1
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = 4, 4, 1
            elif 'downsample.0' in n:
                m.stride = 1

        self.ppm = _PyramidPoolingModule(2048, 512, (2, 4, 6, 8))
        self.final = nn.Sequential(
            nn.Conv3d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(512, num_classes, kernel_size=1)
        )

        initialize_weights(self.ppm, self.final)

    def forward(self, x, spacing):
        x_size = x.size() # 128, 128, 64
        x = self.layer0(x) # 64, 64, 32
        x = self.layer1(x) # 64, 64, 32
        x = self.layer2(x) # 32, 32, 16
        x = self.layer3(x) # 32, 32, 16
        x = self.layer4(x) # 32, 32, 16
        x = self.ppm(x)
        x = self.final(x) # 32, 32, 16
        output = F.upsample(x, x_size[2:], mode='trilinear') # 128 ,128, 64
        output = output.permute(0, 2, 3, 4, 1).contiguous()
        return output.contiguous()


class _FeaturePyramidModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_FeaturePyramidModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool3d(s),  # output zize: sxs
                nn.Conv3d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm3d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)))

        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='trilinear'))
        out = torch.cat(out, 1)
        return out


class PSPUnet(nn.Module):
    def __init__(self, num_classes=1, **kwargs):
        super(PSPUnet, self).__init__()
        self.conv1 = ConvBlock(1, 32, None)
        self.conv2 = ConvBlock(32, 64, None)
        self.conv3_1 = ConvBlock(64, 128, None)
        self.conv3_2 = ConvBlock(128, 128, None)
        self.conv4_1 = ConvBlock(128, 128, None)
        self.conv4_2 = ConvBlock(128, 128, None)
        self.conv5_1 = ConvBlock(128, 128, None)
        # self.conv5_2 = ConvBlock(128, 128, None)
        self.conv5_2 = ConvBlock(128+128+128+64+32, 128, None)

        self.max_pool = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(int(config.TRAIN_PATCH_SIZE[0]/16),
                                                          int(config.TRAIN_PATCH_SIZE[1]/16), int(config.TRAIN_PATCH_SIZE[2])))

        self.deconv0_1 = DeconvBlock(128, 128)
        self.deconv0_2 = ConvBlock(128 + 128, 128, None)
        self.deconv1_1 = DeconvBlock(128, 64)
        self.deconv1_2 = ConvBlock(64 + 128, 64, None)
        self.deconv2_1 = DeconvBlock(64, 32)
        self.deconv2_2 = ConvBlock(32 + 64, 32, None)
        self.deconv3_1 = DeconvBlock(32, 16)
        self.deconv3_2 = ConvBlock(16 + 32, 16, None)
        self.conv8 = nn.Conv3d(16, num_classes, kernel_size=(3, 3, 1), padding=(1, 1, 0))

    def forward(self, x, spacing):
        x_size = x.size() # 128, 128, 64
        conv1 = self.conv1(x)
        pool1 = self.max_pool(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.max_pool(conv2)
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.max_pool(conv3_2)
        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.max_pool(conv4_2)
        conv5_1 = self.conv5_1(pool4)
        lowest_x = torch.cat((self.avg_pool(conv1), self.avg_pool(conv2),
                              self.avg_pool(conv3_1), self.avg_pool(conv4_1), conv5_1), 1)
        conv5_2 = self.conv5_2(lowest_x)

        deconv0_1 = self.deconv0_1(conv5_2)
        deconv0_2 = self.deconv0_2(torch.cat((deconv0_1, conv4_2), 1))
        deconv1_1 = self.deconv1_1(deconv0_2)
        deconv1_2 = self.deconv1_2(torch.cat((deconv1_1, conv3_2), 1))
        deconv2_1 = self.deconv2_1(deconv1_2)
        deconv2_2 = self.deconv2_2(torch.cat((deconv2_1, conv2), 1))
        deconv3_1 = self.deconv3_1(deconv2_2)
        deconv3_2 = self.deconv3_2(torch.cat((deconv3_1, conv1), 1))
        output = self.conv8(deconv3_2)
        output = output.permute(0, 2, 3, 4, 1).contiguous()
        return output.contiguous()


if __name__ == '__main__':
    import time
    import numpy as np

    t = time.time()
    net = PSPUnet()
    x = torch.ones((2, 1, 224, 224, 32))
    spacing = torch.from_numpy(np.asarray([[0.8, 0.8, 0.8], [2.6, 1.2, 2.5]]))
    output = net(x, spacing)
    print(output.size())
    print(time.time() - t)

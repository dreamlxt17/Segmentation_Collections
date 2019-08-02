#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Tencent, Inc. All Rights Reserved
#
"""
File: ASPP.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2019-3-20 16:22:09
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn import functional as F


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


class ASPP(nn.Module):

    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate,bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate,bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate,bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
                nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
#        self.conv_cat = nn.Sequential(
#                nn.Conv2d(dim_out*4, dim_out, 1, 1, padding=0),
#                nn.BatchNorm2d(dim_out),
#                nn.ReLU(inplace=True),
#        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        # global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        global_feature = F.upsample_bilinear(global_feature, (row, col), None)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class SpacingBlock(nn.Module):
    # embedded gaussian
    def __init__(self, in_channels, convblock=ConvBlock, sub_sample=True):
        super(SpacingBlock, self).__init__()
        inter_channels = int(in_channels // 2)

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.pool1 = nn.MaxPool3d(kernel_size=(1,1,2))
        self.conv2 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.pool2 = nn.MaxPool3d(kernel_size=(1,1,2))
        self.conv3 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv3d(inter_channels, in_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(in_channels//4), num_channels=in_channels),
                                   nn.ReLU())

    def forward(self, x, spacings):
        '''
        :param x: (b, c, z, h, w)
        :return:
        '''

        batch_size, c, w, h, z = x.size()
        res = torch.zeros_like(x)
        for i in range(batch_size):
            space = spacings[i][-1]
            y = x[i].view(1, c, w, h, z)
            if space <= 1.5:
                out = self.conv1(y)
                out = self.pool1(out)
                out = self.conv2(out)
                out = self.pool2(out)
                out = self.conv3(out)
                out = F.upsample(out, size=(w, h, z))
                out = self.conv4(out)
            elif space < 3:
                out = self.conv1(y)
                out = self.pool2(out)
                out = self.conv3(out)
                out = F.upsample(out, size=(w, h, z))
                out = self.conv4(out)
            else:
                out = self.conv1(y)
                out = self.conv4(out)
            res[i] = out
        return res


class SpacingBlock1(nn.Module):
    # embedded gaussian
    def __init__(self, in_channels, convblock=ConvBlock, sub_sample=True):
        super(SpacingBlock1, self).__init__()
        inter_channels = int(in_channels * 2)

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.pool1 = nn.MaxPool3d(kernel_size=(1,1,2))
        self.conv2 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.pool2 = nn.MaxPool3d(kernel_size=(1,1,2))
        self.conv3 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv3d(inter_channels, in_channels, kernel_size=1),
                                   nn.GroupNorm(num_groups=int(in_channels//4), num_channels=in_channels),
                                   nn.ReLU())

    def forward(self, x, spacings):
        '''
        :param x: (b, c, z, h, w)

        :return:
        '''

        batch_size, c, w, h, z = x.size()
        res = torch.zeros_like(x)
        for i in range(batch_size):
            space = spacings[i][-1]
            y = x[i].view(1, c, w, h, z)
            if space <= 1.5:
                out = self.conv1(y)
                out = self.pool1(out)
                out = self.conv2(out)
                out = self.pool2(out)
                out = self.conv3(out)
                out = F.upsample(out, size=(w, h, z))
                out = self.conv4(out)
                out = self.conv5(out)
            elif space < 3:
                out = self.conv1(y)
                out = self.pool2(out)
                out = self.conv3(out)
                out = F.upsample(out, size=(w, h, z))
                out = self.conv4(out)
                out = self.conv5(out)
            else:
                out = self.conv1(y)
                out = self.conv4(out)
                out = self.conv5(out)
            res[i] = out
        return res


class SpacingBlock2(nn.Module):
    # embedded gaussian
    def __init__(self, in_channels, convblock=ConvBlock, sub_sample=True):
        super(SpacingBlock2, self).__init__()
        inter_channels = in_channels

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.pool1 = nn.MaxPool3d(kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))
        self.conv2 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.pool2 = nn.MaxPool3d(kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))
        self.conv3 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv3d(inter_channels, in_channels, kernel_size=1),
                                   nn.GroupNorm(num_groups=int(in_channels//4), num_channels=in_channels),
                                   nn.ReLU())

    def forward(self, x, spacings):
        '''
        max-pooling residual
        :param x: (b, c, w, h, z)
        :return:
        '''

        batch_size, c, w, h, z = x.size()
        res = torch.zeros_like(x)
        for i in range(batch_size):
            space = spacings[i][-1]
            y = x[i].view(1, c, w, h, z)
            if space <= 1.5:
                out = self.pool1(y)
                out = self.conv1(out)
                out = self.pool2(out)
                out = self.conv2(out)
                out += y
                out = self.conv4(out)
            elif space < 3:
                out = self.pool2(y)
                out = self.conv1(out)
                out += y
                out = self.conv4(out)
            else:
                out = self.conv1(y)
                out += y
                out = self.conv4(out)
            res[i] = out
        return res


class SpacingBlock3(nn.Module):
    # embedded gaussian
    def __init__(self, in_channels, convblock=ConvBlock, sub_sample=True):
        super(SpacingBlock3, self).__init__()
        inter_channels = in_channels

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, inter_channels, kernel_size=(1, 1, 3), padding=(1, 1, 2), dilation=2),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(1, 1, 3), padding=(1, 1, 4), dilation=4),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(1, 1, 3), padding=(1, 1, 6), dilation=6),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(1, 1, 3), padding=(1, 1, 8), dilation=8),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())

        self.cat_conv1 = nn.Sequential(nn.Conv3d(inter_channels*3+in_channels, in_channels, kernel_size=1),
                                   nn.GroupNorm(num_groups=1, num_channels=in_channels),
                                   nn.ReLU())

        self.cat_conv2 = nn.Sequential(nn.Conv3d(inter_channels*2+in_channels, in_channels, kernel_size=1),
                                   nn.GroupNorm(num_groups=1, num_channels=in_channels),
                                   nn.ReLU())

        self.cat_conv3 = nn.Sequential(nn.Conv3d(inter_channels+in_channels, in_channels, kernel_size=1),
                                   nn.GroupNorm(num_groups=1, num_channels=in_channels),
                                   nn.ReLU())

    def forward(self, x, spacings):
        '''
        :param x: (b, c, w, h, z)
        :return:
        '''

        batch_size, c, w, h, z = x.size()
        res = torch.zeros_like(x)
        for i in range(batch_size):
            space = spacings[i][-1]
            y = x[i].view(1, c, w, h, z)
            if space <= 1.5:
                out1 = self.conv1(y)
                out2 = self.conv2(y)
                out3 = self.conv3(y)
                out = torch.cat((out1, out2, out3), dim=1)
                out = self.cat_conv1(out)
            elif space < 3:
                out1 = self.conv1(y)
                out2 = self.conv2(y)
                out = torch.cat((out1, out2), dim=1)
                out = self.cat_conv2(out)
            else:
                out = self.conv1(y)
                out += y
                out = self.conv4(out)
            res[i] = out
        return res


class SpacingBlock4(nn.Module):
    # embedded gaussian
    def __init__(self, in_channels, convblock=ConvBlock, sub_sample=True):
        super(SpacingBlock4, self).__init__()
        inter_channels = in_channels

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.pool1 = nn.MaxPool3d(kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))
        self.conv2 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.pool2 = nn.MaxPool3d(kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))
        self.conv3 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv3d(inter_channels, in_channels, kernel_size=1),
                                   nn.GroupNorm(num_groups=int(in_channels//4), num_channels=in_channels),
                                   nn.ReLU())

    def forward(self, x, spacings):
        '''
        max-pooling residual
        :param x: (b, c, w, h, z)
        :return:
        '''

        batch_size, c, w, h, z = x.size()
        res = torch.zeros_like(x)
        for i in range(batch_size):
            space = spacings[i][-1]
            y = x[i].view(1, c, w, h, z)
            if space <= 1.0:
                out = self.pool1(y)
                out = self.conv1(out)
                out = self.pool2(out)
                out = self.conv2(out)
                out += y
                out = self.conv4(out)
            else:
                out = self.pool2(y)
                out = self.conv1(out)
                out += y
                out = self.conv4(out)
            res[i] = out
        return res


class SpacingBlock5(nn.Module):
    def __init__(self, in_channels, convblock=ConvBlock, sub_sample=True):
        super(SpacingBlock5, self).__init__()
        inter_channels = in_channels

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.pool1 = nn.MaxPool3d(kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))
        self.conv2 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.pool2 = nn.MaxPool3d(kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))
        self.conv3 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
                                   nn.GroupNorm(num_groups=int(inter_channels//4), num_channels=inter_channels),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv3d(inter_channels, in_channels, kernel_size=1),
                                   nn.GroupNorm(num_groups=int(in_channels//4), num_channels=in_channels),
                                   nn.ReLU())

    def forward(self, x, spacings):
        '''
        max-pooling residual
        :param x: (b, c, w, h, z)
        :return:
        '''

        batch_size, c, w, h, z = x.size()
        res = torch.zeros_like(x)
        for i in range(batch_size):
            space = spacings[i][-1]
            y = x[i].view(1, c, w, h, z)
            if space <= 1.0:
                out = self.pool1(y)
                out = self.conv1(out)
                out = self.pool2(out)
                out = self.conv2(out)
                out += y
                out = self.conv4(out)
            else:
                out = self.pool2(y)
                out = self.conv1(out)
                out += y
                out = self.conv4(out)
            res[i] = out
        return res

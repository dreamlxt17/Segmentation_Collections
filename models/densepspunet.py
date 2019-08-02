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
from .unet3d import ConvBlock, DeconvBlock
from tumor_config import Config
config = Config()


class DensePspUnet(nn.Module):
    def __init__(self, num_classes=1, config=Config, **kwargs):
        super(DensePspUnet, self).__init__()
        W, H, Z = config.TRAIN_PATCH_SIZE
        self.conv1 = ConvBlock(1, 32, None)
        self.conv2 = ConvBlock(32, 64, None)
        self.conv3_1 = ConvBlock(64, 128, None)
        self.conv3_2 = ConvBlock(128, 128, None)
        self.conv4_1 = ConvBlock(128, 128, None)
        self.conv4_2 = ConvBlock(128, 128, None)
        self.conv5_1 = ConvBlock(128, 128, None)
        self.conv5_2 = ConvBlock(128, 128, None)

        self.max_pool = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.max_pool_4 = nn.AdaptiveMaxPool3d(output_size=(int(W/8), int(H/8), int(Z)))
        self.max_pool_3 = nn.AdaptiveMaxPool3d(output_size=(int(W/4), int(H/4), int(Z)))
        self.max_pool_2 = nn.AdaptiveMaxPool3d(output_size=(int(W/2), int(H/2), int(Z)))

        self.deconv0_1 = DeconvBlock(128, 128)
        self.deconv0_2 = ConvBlock(128 + 128 + 32 + 64 + 128, 128, None)
        self.deconv1_1 = DeconvBlock(128, 64)
        self.deconv1_2 = ConvBlock(64 + 128 + 32 +64, 64, None)
        self.deconv2_1 = DeconvBlock(64, 32)
        self.deconv2_2 = ConvBlock(32 + 64 + 32, 32, None)
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
        conv5_2 = self.conv5_2(conv5_1)

        deconv0_1 = self.deconv0_1(conv5_2)
        cat_feature = torch.cat((deconv0_1, conv4_2, self.max_pool_4(conv1), self.max_pool_4(conv2),
                                 self.max_pool_4(conv3_1)), dim=1)
        deconv0_2 = self.deconv0_2(cat_feature)
        deconv1_1 = self.deconv1_1(deconv0_2)
        cat_feature = torch.cat((deconv1_1, conv3_2, self.max_pool_3(conv1), self.max_pool_3(conv2)), dim=1)
        deconv1_2 = self.deconv1_2(cat_feature)
        deconv2_1 = self.deconv2_1(deconv1_2)
        cat_feature = torch.cat((deconv2_1, conv2, self.max_pool_2(conv1)), dim=1)
        deconv2_2 = self.deconv2_2(cat_feature)
        deconv3_1 = self.deconv3_1(deconv2_2)
        deconv3_2 = self.deconv3_2(torch.cat((deconv3_1, conv1), 1))
        output = self.conv8(deconv3_2)
        output = output.permute(0, 2, 3, 4, 1).contiguous()
        return output.contiguous()


class DensePspUnetV2(nn.Module):
    def __init__(self, num_classes=1, config=Config, **kwargs):
        super(DensePspUnetV2, self).__init__()
        W, H, Z = config.TRAIN_PATCH_SIZE
        self.conv1 = ConvBlock(1, 32, None)
        self.conv2 = ConvBlock(32, 64, None)
        self.conv3_1 = ConvBlock(64, 128, None)
        self.conv3_2 = ConvBlock(128, 128, None)
        self.conv4_1 = ConvBlock(128, 128, None)
        self.conv4_2 = ConvBlock(128, 128, None)
        self.conv5_1 = ConvBlock(128, 128, None)
        self.conv5_2 = ConvBlock(128, 128, None)

        self.max_pool = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.max_pool_4 = nn.AdaptiveMaxPool3d(output_size=(int(W/8), int(H/8), int(Z)))
        self.max_pool_3 = nn.AdaptiveMaxPool3d(output_size=(int(W/4), int(H/4), int(Z)))
        self.max_pool_2 = nn.AdaptiveMaxPool3d(output_size=(int(W/2), int(H/2), int(Z)))

        self.deconv0_1 = DeconvBlock(128, 128)
        self.deconv0_2 = ConvBlock(128 + 128 + 32 + 64 + 128, 128, None)
        self.deconv1_1 = DeconvBlock(128, 64)
        self.deconv1_2 = ConvBlock(64 + 128 + 32 + 64 + 128, 64, None)
        self.deconv2_1 = DeconvBlock(64, 32)
        self.deconv2_2 = ConvBlock(32 + 64 + 32 + 128 + 128, 64, None)
        self.deconv3_1 = DeconvBlock(64, 16)
        self.deconv3_2 = ConvBlock(16 + 32 + 64, 16, None)
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
        conv5_2 = self.conv5_2(conv5_1)

        deconv0_1 = self.deconv0_1(conv5_2)
        cat_feature = torch.cat((deconv0_1, conv4_2, self.max_pool_4(conv1), self.max_pool_4(conv2),
                                 self.max_pool_4(conv3_1)), dim=1)
        deconv0_2 = self.deconv0_2(cat_feature)
        deconv1_1 = self.deconv1_1(deconv0_2)
        _, _, w, h, z = deconv1_1.size()
        cat_feature = torch.cat((deconv1_1, conv3_2, self.max_pool_3(conv1), self.max_pool_3(conv2),
                                F.upsample(conv4_2, size=(w, h, z), mode='trilinear')), dim=1)
        deconv1_2 = self.deconv1_2(cat_feature)
        deconv2_1 = self.deconv2_1(deconv1_2)
        _, _, w, h, z = deconv2_1.size()
        cat_feature = torch.cat((deconv2_1, conv2, self.max_pool_2(conv1), F.upsample(conv4_2, size=(w, h, z), mode='trilinear'),
                                 F.upsample(conv3_2, size=(w, h, z), mode='trilinear')), dim=1)
        deconv2_2 = self.deconv2_2(cat_feature)
        deconv3_1 = self.deconv3_1(deconv2_2)
        _, _, w, h, z = deconv3_1.size()
        cat_feature = torch.cat((deconv3_1, conv1, F.upsample(conv2, size=(w, h, z), mode='trilinear')), dim=1)
        deconv3_2 = self.deconv3_2(cat_feature)
        output = self.conv8(deconv3_2)
        output = output.permute(0, 2, 3, 4, 1).contiguous()
        return output.contiguous()


class DensePspUnetV3(nn.Module):
    def __init__(self, num_classes=1, config=Config, **kwargs):
        super(DensePspUnetV3, self).__init__()
        W, H, Z = config.TRAIN_PATCH_SIZE
        self.conv1 = ConvBlock(1, 32, None)
        self.conv2 = ConvBlock(32, 64, None)
        self.conv3_1 = ConvBlock(64, 128, None)
        self.conv3_2 = ConvBlock(128, 128, None)
        self.conv4_1 = ConvBlock(128, 128, None)
        self.conv4_2 = ConvBlock(128, 128, None)
        self.conv5_1 = ConvBlock(128, 128, None)
        self.conv5_2 = ConvBlock(128, 128, None)

        self.max_pool = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.max_pool_4 = nn.AdaptiveMaxPool3d(output_size=(int(W/8), int(H/8), int(Z)))
        self.max_pool_3 = nn.AdaptiveMaxPool3d(output_size=(int(W/4), int(H/4), int(Z)))
        self.max_pool_2 = nn.AdaptiveMaxPool3d(output_size=(int(W/2), int(H/2), int(Z)))

        self.deconv0_1 = DeconvBlock(128, 128)
        self.deconv0_2 = ConvBlock(128 + 128 + 128, 128, None)
        self.deconv1_1 = DeconvBlock(128, 64)
        self.deconv1_2 = ConvBlock(64 + 128 + 64, 64, None)
        self.deconv2_1 = DeconvBlock(64, 32)
        self.deconv2_2 = ConvBlock(32 + 64 + 32, 32, None)
        self.deconv3_1 = DeconvBlock(32, 16)
        self.deconv3_2 = ConvBlock(16 + 32, 16, None)
        self.conv8 = nn.Conv3d(16, num_classes, kernel_size=(3, 3, 1), padding=(1, 1, 0))

    def forward(self, x, spacing):
        x_size = x.size()
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
        conv5_2 = self.conv5_2(conv5_1)

        deconv0_1 = self.deconv0_1(conv5_2)
        cat_feature = torch.cat((deconv0_1, conv4_2,
                                 self.max_pool_4(conv3_1)), dim=1)
        deconv0_2 = self.deconv0_2(cat_feature)
        deconv1_1 = self.deconv1_1(deconv0_2)
        _, _, w, h, z = deconv1_1.size()
        cat_feature = torch.cat((deconv1_1, conv3_2, self.max_pool_3(conv2),
                                ), dim=1)
        deconv1_2 = self.deconv1_2(cat_feature)
        deconv2_1 = self.deconv2_1(deconv1_2)
        _, _, w, h, z = deconv2_1.size()
        cat_feature = torch.cat((deconv2_1, conv2, self.max_pool_2(conv1)), dim=1)
        deconv2_2 = self.deconv2_2(cat_feature)
        deconv3_1 = self.deconv3_1(deconv2_2)
        _, _, w, h, z = deconv3_1.size()
        cat_feature = torch.cat((deconv3_1, conv1), dim=1)
        deconv3_2 = self.deconv3_2(cat_feature)
        output = self.conv8(deconv3_2)
        output = output.permute(0, 2, 3, 4, 1).contiguous()
        return output.contiguous()


class DensePspUnetV4(nn.Module):
    def __init__(self, num_classes=1, config=Config, **kwargs):
        super(DensePspUnetV4, self).__init__()
        W, H, Z = config.TRAIN_PATCH_SIZE
        self.conv1 = ConvBlock(1, 32, None)
        self.conv2 = ConvBlock(32, 64, None)
        self.conv3_1 = ConvBlock(64, 128, None)
        self.conv3_2 = ConvBlock(128, 128, None)
        self.conv4_1 = ConvBlock(128, 128, None)
        self.conv4_2 = ConvBlock(128, 128, None)
        self.conv5_1 = ConvBlock(128, 128, None)
        self.conv5_2 = ConvBlock(128, 128, None)

        self.max_pool = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.deconv0_1 = DeconvBlock(128, 128)
        self.deconv0_2 = ConvBlock(128 + 128, 128, None)
        self.deconv1_1 = DeconvBlock(128, 64)
        self.deconv1_2 = ConvBlock(64 + 128 + 128, 64, None)
        self.deconv2_1 = DeconvBlock(64, 32)
        self.deconv2_2 = ConvBlock(32 + 64 + 128, 64, None)
        self.deconv3_1 = DeconvBlock(64, 16)
        self.deconv3_2 = ConvBlock(16 + 32 + 64, 16, None)
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
        conv5_2 = self.conv5_2(conv5_1)

        deconv0_1 = self.deconv0_1(conv5_2)
        cat_feature = torch.cat((deconv0_1, conv4_2), dim=1)
        deconv0_2 = self.deconv0_2(cat_feature)
        deconv1_1 = self.deconv1_1(deconv0_2)
        _, _, w, h, z = deconv1_1.size()
        cat_feature = torch.cat((deconv1_1, conv3_2,
                                F.upsample(conv4_2, size=(w, h, z), mode='trilinear')), dim=1)
        deconv1_2 = self.deconv1_2(cat_feature)
        deconv2_1 = self.deconv2_1(deconv1_2)
        _, _, w, h, z = deconv2_1.size()
        cat_feature = torch.cat((deconv2_1, conv2, F.upsample(conv3_2, size=(w, h, z), mode='trilinear')), dim=1)
        deconv2_2 = self.deconv2_2(cat_feature)
        deconv3_1 = self.deconv3_1(deconv2_2)
        _, _, w, h, z = deconv3_1.size()
        cat_feature = torch.cat((deconv3_1, conv1, F.upsample(conv2, size=(w, h, z), mode='trilinear')), dim=1)
        deconv3_2 = self.deconv3_2(cat_feature)
        output = self.conv8(deconv3_2)
        output = output.permute(0, 2, 3, 4, 1).contiguous()
        return output.contiguous()


class DensePspUnetV4_1(nn.Module):
    def __init__(self, num_classes=1, config=Config, **kwargs):
        super(DensePspUnetV4_1, self).__init__()
        W, H, Z = config.TRAIN_PATCH_SIZE
        self.conv1 = ConvBlock(1, 32, None)
        self.conv2 = ConvBlock(32, 64, None)
        self.conv3_1 = ConvBlock(64, 128, None)
        self.conv3_2 = ConvBlock(128, 128, None)
        self.conv4_1 = ConvBlock(128, 128, None)
        self.conv4_2 = ConvBlock(128, 128, None)
        self.conv5_1 = ConvBlock(128, 128, None)
        self.conv5_2 = ConvBlock(128, 128, None)

        self.max_pool = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.deconv0_1 = DeconvBlock(128, 128)
        self.deconv0_2 = ConvBlock(128 + 128, 128)
        self.deconv0_3 = ConvBlock(128, 128, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.deconv1_1 = DeconvBlock(128, 64)
        self.deconv1_2 = ConvBlock(64 + 128 + 128, 64)
        self.deconv1_3 = ConvBlock(64, 64, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.deconv2_1 = DeconvBlock(64, 32)
        self.deconv2_2 = ConvBlock(32 + 64 + 128, 64)
        self.deconv2_3 = ConvBlock(64, 64, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.deconv3_1 = DeconvBlock(64, 16)
        self.deconv3_2 = ConvBlock(16 + 32 + 64, 16)
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
        conv5_2 = self.conv5_2(conv5_1)

        deconv0_1 = self.deconv0_1(conv5_2)
        cat_feature = torch.cat((deconv0_1, conv4_2), dim=1)
        deconv0_2 = self.deconv0_2(cat_feature)
        deconv0_3 = self.deconv0_3(deconv0_2)
        deconv1_1 = self.deconv1_1(deconv0_3)
        _, _, w, h, z = deconv1_1.size()
        cat_feature = torch.cat((deconv1_1, conv3_2,
                                F.upsample(conv4_2, size=(w, h, z), mode='trilinear')), dim=1)
        deconv1_2 = self.deconv1_2(cat_feature)
        deconv1_3 = self.deconv1_3(deconv1_2)
        deconv2_1 = self.deconv2_1(deconv1_3)
        _, _, w, h, z = deconv2_1.size()
        cat_feature = torch.cat((deconv2_1, conv2, F.upsample(conv3_2, size=(w, h, z), mode='trilinear')), dim=1)
        deconv2_2 = self.deconv2_2(cat_feature)
        deconv2_3 = self.deconv2_3(deconv2_2)
        deconv3_1 = self.deconv3_1(deconv2_3)
        _, _, w, h, z = deconv3_1.size()
        cat_feature = torch.cat((deconv3_1, conv1, F.upsample(conv2, size=(w, h, z), mode='trilinear')), dim=1)
        deconv3_2 = self.deconv3_2(cat_feature)
        output = self.conv8(deconv3_2)
        output = output.permute(0, 2, 3, 4, 1).contiguous()
        return output.contiguous()


if __name__ == '__main__':
    import time
    import numpy as np

    t = time.time()
    net = DensePspUnetV4_1()
    x = torch.ones((2, 1, 192, 192, 24))
    spacing = torch.from_numpy(np.asarray([[0.8, 0.8, 0.8], [2.6, 1.2, 2.5]]))
    output = net(x, spacing)
    print(output.size())
    print(time.time() - t)

# coding=utf-8
# Author: Didia
# Date: 19-6-10

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class _Separable1(nn.Module):
    # embedded gaussian
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=False, bn_layer=True, spacing_thresh=1.5):
        super(_Separable1, self).__init__()

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.spacing_thresh = spacing_thresh

        if self.inter_channels is None:
            self.inter_channels = in_channels
        if self.inter_channels == 0:
            self.inter_channels = 1

        conv_nd = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 1, 2))
        bn = nn.BatchNorm3d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1), groups=self.in_channels)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, spacings):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size, c, w, h, z = x.size()
        res = torch.zeros_like(x)
        for i in range(batch_size):
            space = spacings[i][-1]
            y = x[i].view(1, c, w, h, z)
            if space <= self.spacing_thresh:
                res[i] = self.compute_x(y)
            else:
                res[i] = x[i]
        return self.W(res)

    def compute_x(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size, c, w, h, z = x.size()
        if self.sub_sample:
            new_z = z/2
        else:
            new_z = z

        # b, w, h, z, c
        g_x = self.g(x).permute(0, 2, 3, 4, 1).contiguous()
        g_x = g_x.view(-1, new_z, self.inter_channels)
        # b, w, h, z, c
        theta_x = self.theta(x).permute(0, 2, 3, 4, 1).contiguous()
        theta_x = theta_x.view(-1, z, self.inter_channels)
        # b, w, h, c, z
        phi_x = self.phi(x).permute(0, 2, 3, 1, 4).contiguous()
        phi_x = phi_x.view(-1, self.inter_channels, new_z)
        # bwh, z, z
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        # bwh, z, c
        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, w, h, z, self.inter_channels).contiguous()
        y = y.permute(0, 4, 1, 2, 3)
        z = y + x

        return z


class Separable1(_Separable1):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=False, bn_layer=True, spacing_thresh=1.5):
        super(Separable1, self).__init__(in_channels, inter_channels=inter_channels, sub_sample=sub_sample,
                                         bn_layer=bn_layer, spacing_thresh=spacing_thresh)


def count_parameter(net):
    # 计算网络参数
    num_parameter = .0
    for item in net.modules():

        if isinstance(item, nn.Conv3d) or isinstance(item, nn.ConvTranspose3d):
            num_parameter += (item.weight.size(0) * item.weight.size(1) *
                              item.weight.size(2) * item.weight.size(3) * item.weight.size(4))

            if item.bias is not None:
                num_parameter += item.bias.size(0)

        elif isinstance(item, nn.PReLU):
            num_parameter += item.num_parameters

    print(num_parameter)


if __name__=='__main__':
    from torch.autograd import Variable
    import numpy as np

    img = Variable(torch.randn(2, 32, 112, 112, 8)).cuda()  #
    spacing = torch.from_numpy(np.asarray([[0.8, 0.8, 0.8], [2.6, 1.2, 2.5]]))
    net = Separable1(in_channels=32, bn_layer=True).cuda()
    out = net(img, spacing)
    print(img.size(), out.size())

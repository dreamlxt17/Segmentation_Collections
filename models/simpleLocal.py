# coding=utf-8
# Author: Didia
# Date: 19-6-3

import torch
from torch import nn
from torch.nn import functional as F


class _NonLocal0(nn.Module):
    # embedded gaussian
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, spacing_thresh=1.5):
        super(_NonLocal0, self).__init__()

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.spacing_thresh = spacing_thresh

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 1, 2))
        bn = nn.BatchNorm3d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

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
            # self.theta = nn.Sequential(self.theta, max_pool_layer)

    def forward(self, x, spacings):
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
        W_y = self.W(y)
        z = W_y + x

        return z


class _NonLocal1(nn.Module):
    # embedded gaussian
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, spacing_thresh=1.5):
        super(_NonLocal1, self).__init__()

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.spacing_thresh = spacing_thresh

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 1, 2))
        bn = nn.BatchNorm3d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal_(self.W.weight)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            # self.theta = nn.Sequential(self.theta, max_pool_layer)

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
        return res

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
        W_y = self.W(y)
        z = W_y + x

        return z


class _NonLocal2(nn.Module):
    # embedded gaussian
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, spacing_thresh=1.5):
        super(_NonLocal2, self).__init__()

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
        # nn.init.kaiming_normal_(conv_nd.weight)

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal_(self.W.weight)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            # self.theta = nn.Sequential(self.theta, max_pool_layer)
        self.pooling = nn.AdaptiveAvgPool3d(output_size=1)

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
        return res

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
        p = self.pooling(x.permute(0, 4, 2, 3, 1)).permute(0, 2, 3, 4, 1)
        # x = x * p
        theta_x = (x * p).permute(0, 2, 3, 4, 1).contiguous()
        theta_x = theta_x.view(-1, z, c)
        # b, w, h, c, z
        phi_x = self.phi(x * p).permute(0, 2, 3, 1, 4).contiguous()
        phi_x = phi_x.view(-1, self.inter_channels, new_z)
        # bwh, z, z
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        # bwh, z, c
        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, w, h, z, self.inter_channels).contiguous()
        y = y.permute(0, 4, 1, 2, 3)
        W_y = self.W(y)
        z = W_y + x

        return z


class _NonLocal3(nn.Module):
    # embedded gaussian
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, spacing_thresh=1.5):
        super(_NonLocal3, self).__init__()

        self.dimension = dimension
        self.spacing_thresh = spacing_thresh
        self.in_channels = in_channels

        conv_nd = nn.Conv3d

        self.v = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)

        self.k = conv_nd(in_channels=self.in_channels, out_channels=1,
                         kernel_size=1, stride=1, padding=0)

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
        return res

    def compute_x(self, x):

        # b, w, h, z, c
        batch_size, c, w, h, z = x.size()
        v_x = self.v(x).permute(0, 2, 3, 1, 4).contiguous()
        v_x = v_x.view(-1, self.in_channels, z)
        k_x = self.k(x).permute(0, 2, 3, 4, 1).contiguous()
        k_x = k_x.view(-1, z, 1)

        k_x = F.softmax(k_x, dim=-1)
        f = torch.matmul(v_x, k_x).squeeze()
        f = f.view(batch_size, w, h, 1, c).permute(0, 4, 1, 2, 3)
        z = torch.add(x, f)

        return z


class _NonLocal4(nn.Module):
    # embedded gaussian
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, spacing_thresh=1.5):
        super(_NonLocal4, self).__init__()

        self.dimension = dimension
        self.spacing_thresh = spacing_thresh
        self.in_channels = in_channels

        conv_nd = nn.Conv3d
        bn = nn.BatchNorm3d

        self.v = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                         kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))

        self.k = conv_nd(in_channels=self.in_channels, out_channels=1,
                         kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal_(self.W.weight)
            nn.init.constant_(self.W.bias, 0)

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
            if space <= self.spacing_thresh:
                res[i] = self.compute_x(y)
            else:
                res[i] = x[i]
        res = self.W(res)
        return res

    def compute_x(self, x):

        # b, w, h, z, c
        _, c, w, h, z = x.size()
        v_x = self.v(x).permute(0, 2, 3, 1, 4).contiguous()
        v_x = v_x.view(-1, self.in_channels, z)
        k_x = self.k(x).permute(0, 2, 3, 4, 1).contiguous()
        k_x = k_x.view(-1, z, 1)

        k_x = F.softmax(k_x, dim=-1)
        f = torch.matmul(v_x, k_x).squeeze()
        f = f.view(1, w, h, 1, c).permute(0, 4, 1, 2, 3)
        z = torch.add(x, f)

        return z


class _NonLocal5(nn.Module):
    # embedded gaussian
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, spacing_thresh=1.5):
        super(_NonLocal5, self).__init__()

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
                         kernel_size=1, stride=1, padding=0)

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
            # self.theta = nn.Sequential(self.theta, max_pool_layer)

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


class _NonLocal7(nn.Module):
        # embedded gaussian
        def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True,
                     spacing_thresh=1.5):
            super(_NonLocal7, self).__init__()

            self.dimension = dimension
            self.sub_sample = sub_sample

            self.in_channels = in_channels
            self.inter_channels = inter_channels
            self.spacing_thresh = spacing_thresh
            self.avgpooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.inter_channels is None:
                self.inter_channels = in_channels // 2
                if self.inter_channels == 0:
                    self.inter_channels = 1

            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 1, 2))
            bn = nn.BatchNorm3d
            self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

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
                # self.theta = nn.Sequential(self.theta, max_pool_layer)

        def forward(self, x, spacings):
            '''
            :param x: (b, c, t, h, w)
            :return:
            '''

            batch_size, c, w, h, z = x.size()
            if self.sub_sample:
                new_z = z / 2
            else:
                new_z = z

            g_x = self.avgpooling(self.g(x).permute(0, 4, 1, 2, 3).contiguous().view(-1, c, w, h))
            g_x = g_x.permute(0, 2, 3, 1).view(batch_size, self.inter_channels, new_z).permute(0, 2, 1)

            theta_x = self.avgpooling(self.theta(x).permute(0, 4, 1, 2, 3).contiguous().view(-1, c, w, h))
            theta_x = theta_x.permute(0, 2, 3, 1).view(batch_size, self.inter_channels, z).permute(0, 2, 1)

            phi_x = self.avgpooling(self.phi(x).permute(0, 4, 1, 2, 3).contiguous().view(-1, c, w, h))
            phi_x = phi_x.permute(0, 2, 3, 1).view(batch_size, self.inter_channels, new_z)

            f = torch.matmul(theta_x, phi_x)
            f_div_C = F.softmax(f, dim=-1)
            # bwh, z, c
            y = torch.matmul(f_div_C, g_x)
            y = y.view(batch_size, 1, 1, z, self.inter_channels).contiguous()
            y = y.permute(0, 4, 1, 2, 3)
            W_y = self.W(y)
            z = W_y + x

            return z


class NonLocal1(_NonLocal1):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=False, spacing_thresh=1.5):
        super(NonLocal1, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer, spacing_thresh=spacing_thresh)


class NonLocal2(_NonLocal2):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=False, spacing_thresh=1.5):
        super(NonLocal2, self).__init__(in_channels,
                                            inter_channels=inter_channels,
                                            dimension=3, sub_sample=sub_sample,
                                            bn_layer=bn_layer, spacing_thresh=spacing_thresh)


class NonLocal3(_NonLocal3):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=False):
        super(NonLocal3, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NonLocal4(_NonLocal4):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=False, spacing_thresh=1.5):
        super(NonLocal4, self).__init__(in_channels,
                                            inter_channels=inter_channels,
                                            dimension=3, sub_sample=sub_sample,
                                            bn_layer=bn_layer, spacing_thresh=spacing_thresh)


class NonLocal5(_NonLocal5):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=False, spacing_thresh=1.5):
        super(NonLocal5, self).__init__(in_channels,
                                            inter_channels=inter_channels,
                                            dimension=3, sub_sample=sub_sample,
                                            bn_layer=bn_layer, spacing_thresh=spacing_thresh)


class NonLocal6(_NonLocal1):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, spacing_thresh=1.5):
        super(NonLocal6, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer, spacing_thresh=spacing_thresh)


class NonLocal0(_NonLocal0):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, spacing_thresh=1.5):
        super(NonLocal0, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer, spacing_thresh=spacing_thresh)


class NonLocal7(_NonLocal7):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, spacing_thresh=1.5):
        super(NonLocal7, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer, spacing_thresh=spacing_thresh)


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch
    import numpy as np

    sub_sample = True
    bn_layer = False

    img = Variable(torch.randn(2, 32, 112, 112, 8)) #
    spacing = torch.from_numpy(np.asarray([[0.8, 0.8, 0.8], [2.6, 1.2, 2.5]]))
    net = NonLocal7(32, inter_channels=None, bn_layer=True)
    out = net(img, spacing)
    print(img.size(), out.size())
import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlock(nn.Module):
    # embedded gaussian
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlock, self).__init__()

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

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
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            # self.theta = nn.Sequential(self.theta, max_pool_layer)

    def forward(self, x):
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


class _NonLocalBlock2(nn.Module):
    # dot product
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlock2, self).__init__()

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

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
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            # self.theta = nn.Sequential(self.theta, max_pool_layer)

    def forward(self, x):
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
        N = f.size(-1)
        f_div_C = f / N
        # bwh, z, c
        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, w, h, z, self.inter_channels).contiguous()
        y = y.permute(0, 4, 1, 2, 3)
        W_y = self.W(y)
        z = W_y + x

        return z


class _NonLocalBlock3(nn.Module):
    # concat
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlock3, self).__init__()

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

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
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            # self.theta = nn.Sequential(self.theta, max_pool_layer)

        self.concat_ = nn.Sequential(nn.Conv2d(self.inter_channels*2, 1, 1, 1, 0, bias=False),
                                     nn.ReLU())

    def forward(self, x):
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
        theta_x = self.theta(x).permute(0, 2, 3, 1, 4).contiguous()
        theta_x = theta_x.view(-1, self.inter_channels, z, 1)
        # b, w, h, c, z
        phi_x = self.phi(x).permute(0, 2, 3, 1, 4).contiguous()
        phi_x = phi_x.view(-1, self.inter_channels, 1, new_z)
        # bwh, z, z
        theta_x = theta_x.repeat(1, 1, 1, phi_x.size(3))
        phi_x = phi_x.repeat(1, 1, theta_x.size(2), 1)

        concat_ = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_(concat_)
        b, c, h1 ,w1 = f.size()
        f = f.view(b, h1, w1)
        N = f.size(-1)
        f_div_C = f / N
        # bwh, z, c
        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, w, h, z, self.inter_channels).contiguous()
        y = y.permute(0, 4, 1, 2, 3)
        W_y = self.W(y)
        z = W_y + x

        return z


class NonLocalBlock(_NonLocalBlock):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=False):
        super(NonLocalBlock, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NonLocalBlock2(_NonLocalBlock2):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=False):
        super(NonLocalBlock2, self).__init__(in_channels, inter_channels, dimension=3,
                                                 sub_sample=sub_sample, bn_layer=bn_layer)


class NonLocalBlock3(_NonLocalBlock3):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=False):
        super(NonLocalBlock3, self).__init__(in_channels, inter_channels, dimension=3,
                                                 sub_sample=sub_sample, bn_layer=bn_layer)


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    sub_sample = True
    bn_layer = False

    img = Variable(torch.randn(1, 32, 20, 20, 10)) #
    net = NonLocalBlock3(32, sub_sample=True, bn_layer=bn_layer)
    out = net(img)
    print(img.size(), out.size())





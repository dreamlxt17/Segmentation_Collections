# coding=utf-8
# Author: Didia
# Date: 18-5-17
import torch
from torch import nn
import torch.nn.functional as F


def conv3X3(in_channels, out_channels, stride=1, pad=1):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=pad, bias=False)


def conv1x1(in_channels, out_channels, stride=1, pad=1):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=pad, bias=False)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3X3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3X3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsampel = downsample
        self.drop = nn.Dropout(p=0.8)
        # self.fc1 = nn.Linear(,2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampel:
            residual = self.downsampel(x)
        out += residual
        # out = self.drop(out)
        return self.relu(out)


class UnetSegment(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(UnetSegment, self).__init__()

        self.in_channels = in_channels
        self.block = ResBlock
        self.attention = Attention
        self.d_layer1 = self.make_layer(1, 32)
        self.d_layer2 = self.make_layer(2, 64)
        self.d_layer3 = self.make_layer(3, 128)
        self.d_layer4 = self.make_layer(4, 256)
        self.d_layer5 = self.make_layer(5, 512)

        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.deconv4 = self.de_conv(512, 256)
        self.u_layer4 = self.make_layer(4+2, 256)
        self.deconv3 = self.de_conv(256, 128)
        self.u_layer3 = self.make_layer(3+2, 128)
        self.deconv2 = self.de_conv(128, 64)
        self.u_layer2 = self.make_layer(2+2, 64)
        self.deconv1 = self.de_conv(64, 32)
        self.u_layer1 = self.make_layer(1+2, 32)

        self.atten4 = self.attention(256, 512)
        self.atten3 =self.attention(128, 256)
        self.atten2 =self.attention(64, 128)
        self.atten1 =self.attention(32, 64)

        self.conv = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)

    def make_layer(self, layer, out_channels, stride=1):
        if layer > 1:
            self.in_channels = 2**(layer+3)
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(conv3X3(self.in_channels, out_channels, stride=stride),
                                       nn.BatchNorm2d(out_channels))
        return self.block(self.in_channels, out_channels, stride, downsample)

    def de_conv(self, in_channels, out_channels, stride=2, pad=1):
        return  nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size=3, stride=stride, padding=pad, output_padding=1, bias=False)

    def max_pool(self, kernel_size=2):
        return nn.MaxPool2d(kernel_size)

    def forward(self, x):
        out1 = self.d_layer1(x)
        out1_p = self.pooling(out1)
        out2 = self.d_layer2(out1_p)
        out2_p = self.pooling(out2)
        out3 = self.d_layer3(out2_p)
        out3_p = self.pooling(out3)
        out4 = self.d_layer4(out3_p)
        # print out4.size(1)
        out4_p = self.pooling(out4)
        out5 = self.d_layer5(out4_p)
        up4 = self.deconv4(out5)
        upout4 = self.u_layer4(torch.cat((out4, up4), 1))
        up3 = self.deconv3(upout4)
        upout3 = self.u_layer3(torch.cat((out3, up3), 1))
        up2 = self.deconv2(upout3)
        upout2 = self.u_layer2(torch.cat((out2, up2), 1))
        up1 = self.deconv1(upout2)
        upout1 = self.u_layer1(torch.cat((out1, up1), 1))

        return self.conv(upout1)



class Attention(nn.Module):
    '''把concat变成attention'''
    def __init__(self, in_channels, out_channels, stride=1):
        super(Attention, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.theta = conv1x1(in_channels, in_channels, pad=0)
        self.phi = conv1x1(out_channels, in_channels, pad=0)
        self.psi = conv1x1(in_channels, 1, pad=0)

        self.W = nn.Sequential(
            conv1x1(in_channels,in_channels, pad=0),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x, g):
        input_size = x.size()
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x+phi_g, inplace=True)
        sigm_psi_f = F.sigmoid(self.psi(f))

        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='bilinear')
        y = sigm_psi_f.expand_as(x)*x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class UNet(UnetSegment):
    def __init__(self, in_channels, out_channels, stride=1):
        super(UNet, self).__init__(in_channels, out_channels)


class AttentionUNet(UnetSegment):
    def __init__(self, in_channels, out_channels, stride=1):
        super(AttentionUNet, self).__init__(in_channels, out_channels)

    def forward(self, x):
        out1 = self.d_layer1(x)
        out1_p = self.pooling(out1)
        out2 = self.d_layer2(out1_p)
        out2_p = self.pooling(out2)
        out3 = self.d_layer3(out2_p)
        out3_p = self.pooling(out3)
        out4 = self.d_layer4(out3_p)
        out4_p = self.pooling(out4)
        out5 = self.d_layer5(out4_p)  # 最底层输出
        up4 = self.deconv4(out5)
        out4, _ = self.atten4(out4, out5)
        upout4 = self.u_layer4(torch.cat((out4, up4), 1))
        up3 = self.deconv3(upout4)
        out3, _ = self.atten3(out3, upout4)
        upout3 = self.u_layer3(torch.cat((out3, up3), 1))
        up2 = self.deconv2(upout3)
        out2, _ = self.atten2(out2, upout3)
        upout2 = self.u_layer2(torch.cat((out2, up2), 1))
        up1 = self.deconv1(upout2)
        out1, _ = self.atten1(out1, upout2)
        upout1 = self.u_layer1(torch.cat((out1, up1), 1))

        return self.conv(upout1), _

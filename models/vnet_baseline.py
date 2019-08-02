# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from spacingBlock import SpacingBlock2, SpacingBlock4


class _Conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1), p=0.3):
        super(_Conv_bn_relu, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel, stride=stride, padding=padding)
        # self.gn = nn.GroupNorm(num_groups=int(out_channels/4), num_channels=out_channels)
        self.gn = nn.GroupNorm(num_groups=16, num_channels=out_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout3d(p=p)

        nn.init.kaiming_normal_(self.conv.weight)
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()
        self.gn.weight.data.fill_(1)
        self.gn.bias.data.zero_()

    def forward(self, x):
        #return self.drop(self.relu(self.gn(self.conv(x))))
        return self.relu(self.gn(self.conv(x)))


class _Deconv_relu(nn.Module):
    def __init__(self, in_channels, out_channels, p=0.3, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1)):
        super(_Deconv_relu, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel, stride=stride, padding=padding)
        self.gn = nn.GroupNorm(num_groups=16, num_channels=out_channels)
        self.relu = nn.ReLU()
        self.drop=nn.Dropout3d(p)

        nn.init.kaiming_normal_(self.conv.weight)
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()
        self.gn.weight.data.fill_(1)
        self.gn.bias.data.zero_()

    def forward(self, x):
        #return self.drop(self.relu(self.conv(x)))
        return self.relu(self.conv(x))


class VNet(nn.Module):
    def __init__(self, num_classes=1, spacingblock=SpacingBlock2, drop_rate=0.3, training=True):
        super(VNet, self).__init__()
        self.training = training
        # print(self.training)
        self.p = drop_rate
        self.layer0 = _Conv_bn_relu(in_channels=1, out_channels=16)
        self.layer1 = _Conv_bn_relu(in_channels=16, out_channels=16, )

        self.down1 = _Conv_bn_relu(in_channels=16, out_channels=32, stride=(2,2,2), )
        self.layer2_1 = _Conv_bn_relu(in_channels=32, out_channels=32, )
        self.layer2_2 = _Conv_bn_relu(in_channels=32, out_channels=32, )

        self.down2 = _Conv_bn_relu(in_channels=32, out_channels=64, stride=(2,2,2), )
        self.layer3_1 = _Conv_bn_relu(in_channels=64, out_channels=64, )
        self.layer3_2= _Conv_bn_relu(in_channels=64, out_channels=64, )
        self.layer3_3= _Conv_bn_relu(in_channels=64, out_channels=64, )

        self.down3 = _Conv_bn_relu(in_channels=64, out_channels=128, stride=(2,2,2), )
        self.layer4_1 = _Conv_bn_relu(in_channels=128, out_channels=128, )
        self.layer4_2 = _Conv_bn_relu(in_channels=128, out_channels=128, )
        self.layer4_3 = _Conv_bn_relu(in_channels=128, out_channels=128, )

        self.down4 = _Conv_bn_relu(in_channels=128, out_channels=256, stride=(2,2,2), )
        self.layer5_1 = _Conv_bn_relu(in_channels=256, out_channels=256, )
        self.layer5_2 = _Conv_bn_relu(in_channels=256, out_channels=256, )
        self.layer5_3 = _Conv_bn_relu(in_channels=256, out_channels=256, )

        self.deconv1 = _Deconv_relu(in_channels=256, out_channels=128, kernel=2, stride=2, padding=0) # different from in tensorflow
        self.layer6_1 = _Conv_bn_relu(in_channels=256, out_channels=128, )
        self.layer6_2 = _Conv_bn_relu(in_channels=128, out_channels=128, )
        self.layer6_3 = _Conv_bn_relu(in_channels=128, out_channels=128, )

        self.deconv2 = _Deconv_relu(in_channels=128, out_channels=64, kernel=2, stride=2, padding=0)
        self.layer7_1 = _Conv_bn_relu(in_channels=128, out_channels=64, )
        self.layer7_2 = _Conv_bn_relu(in_channels=64, out_channels=64, )
        self.layer7_3 = _Conv_bn_relu(in_channels=64, out_channels=64, )

        self.deconv3 = _Deconv_relu(in_channels=64, out_channels=32, kernel=2, stride=2, padding=0)
        self.layer8_1 = _Conv_bn_relu(in_channels=64, out_channels=32, )
        self.layer8_2 = _Conv_bn_relu(in_channels=32, out_channels=32, )
        self.layer8_3 = _Conv_bn_relu(in_channels=32, out_channels=32, )

        self.deconv4 = _Deconv_relu(in_channels=32, out_channels=16, kernel=2, stride=2, padding=0)
        self.layer9_1 = _Conv_bn_relu(in_channels=32, out_channels=16, )
        self.layer9_2 = _Conv_bn_relu(in_channels=16, out_channels=16, )
        self.layer9_3 = _Conv_bn_relu(in_channels=16, out_channels=16, )

        self.output = nn.Conv3d(in_channels=16, out_channels=num_classes, kernel_size=1)

    def forward(self, x, spacings):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer1 += layer0

        down1 = self.down1(layer1)
        layer2 = self.layer2_1(down1)
        layer2 = self.layer2_2(layer2)
        layer2 += down1

        down2 = self.down2(layer2)
        layer3 = self.layer3_1(down2)
        layer3 = self.layer3_2(layer3)
        layer3 = self.layer3_3(layer3)
        layer3 += down2

        down3 = self.down3(layer3)
        layer4 = self.layer4_1(down3)
        layer4 = self.layer4_2(layer4)
        layer4 = self.layer4_3(layer4)
        layer4 += down3

        down4 = self.down4(layer4)
        layer5 = self.layer5_1(down4)
        layer5 = self.layer5_2(layer5)
        layer5 = self.layer5_3(layer5)
        layer5 += down4

        deconv1 = self.deconv1(layer5)
        layer6 = self.layer6_1(torch.cat([deconv1, layer4], dim=1))
        layer6 = self.layer6_2(layer6)
        layer6 = self.layer6_3(layer6)
        layer6 += deconv1

        deconv2 = self.deconv2(layer6)
        layer7 = self.layer7_1(torch.cat([deconv2, layer3], dim=1))
        layer7 = self.layer7_2(layer7)
        layer7 = self.layer7_3(layer7)
        layer7 += deconv2

        deconv3 = self.deconv3(layer7)
        layer8 = self.layer8_1(torch.cat([deconv3, layer2], dim=1))
        layer8 = self.layer8_2(layer8)
        layer8 = self.layer8_3(layer8)
        layer8 += deconv3

        deconv4 = self.deconv4(layer8)
        layer9 = self.layer9_1(torch.cat([deconv4, layer1], dim=1))
        layer9 = self.layer9_2(layer9)
        layer9 = self.layer9_3(layer9)
        layer9 += deconv4

        output = self.output(layer9)
        output = output.permute(0, 2, 3, 4, 1).contiguous()

        return output


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal(module.weight.data, 0.25)
        nn.init.constant(module.bias.data, 0)


class VNet00(VNet):
    def __init__(self, num_classes=1, training=True, **kwargs):
        super(VNet00, self).__init__(num_classes=num_classes, drop_rate=0.2, training=training)


class VNetSimple(VNet):

    def __init__(self, num_classes=1, spacingblock=SpacingBlock2, drop_rate=0.3, training=True):
        super(VNet, self).__init__()
        self.training = training
        # print(self.training)
        self.p = drop_rate
        self.layer0 = _Conv_bn_relu(in_channels=1, out_channels=16)
        self.layer1 = _Conv_bn_relu(in_channels=16, out_channels=16, )

        self.down1 = _Conv_bn_relu(in_channels=16, out_channels=32, stride=(2,2,2), )
        self.layer2_1 = _Conv_bn_relu(in_channels=32, out_channels=32, )
        self.layer2_2 = _Conv_bn_relu(in_channels=32, out_channels=32, )

        self.down2 = _Conv_bn_relu(in_channels=32, out_channels=64, stride=(2,2,2), )
        self.layer3_1 = _Conv_bn_relu(in_channels=64, out_channels=64, )
        self.layer3_2= _Conv_bn_relu(in_channels=64, out_channels=64, )
        self.layer3_3= _Conv_bn_relu(in_channels=64, out_channels=64, )

        self.down3 = _Conv_bn_relu(in_channels=64, out_channels=128, stride=(2,2,2), )
        self.layer4_1 = _Conv_bn_relu(in_channels=128, out_channels=128, )
        self.layer4_2 = _Conv_bn_relu(in_channels=128, out_channels=128, )
        self.layer4_3 = _Conv_bn_relu(in_channels=128, out_channels=128, )

        self.down4 = _Conv_bn_relu(in_channels=128, out_channels=256, stride=(2,2,2), )
        self.layer5_1 = _Conv_bn_relu(in_channels=256, out_channels=256, )
        self.layer5_2 = _Conv_bn_relu(in_channels=256, out_channels=256, )
        self.layer5_3 = _Conv_bn_relu(in_channels=256, out_channels=256, )

        self.deconv1 = _Deconv_relu(in_channels=256, out_channels=128, kernel=2, stride=2, padding=0) # different from in tensorflow
        self.layer6_1 = _Conv_bn_relu(in_channels=256, out_channels=128, )
        self.layer6_2 = _Conv_bn_relu(in_channels=128, out_channels=128, )
        self.layer6_3 = _Conv_bn_relu(in_channels=128, out_channels=128, )

        self.deconv2 = _Deconv_relu(in_channels=128, out_channels=64, kernel=2, stride=2, padding=0)
        self.layer7_1 = _Conv_bn_relu(in_channels=128, out_channels=64, )
        self.layer7_2 = _Conv_bn_relu(in_channels=64, out_channels=64, )
        self.layer7_3 = _Conv_bn_relu(in_channels=64, out_channels=64, )

        self.deconv3 = _Deconv_relu(in_channels=64, out_channels=32, kernel=2, stride=2, padding=0)
        self.layer8_1 = _Conv_bn_relu(in_channels=64, out_channels=32, )
        self.layer8_2 = _Conv_bn_relu(in_channels=32, out_channels=32, )
        self.layer8_3 = _Conv_bn_relu(in_channels=32, out_channels=32, )

        self.deconv4 = _Deconv_relu(in_channels=32, out_channels=16, kernel=2, stride=2, padding=0)
        self.layer9_1 = _Conv_bn_relu(in_channels=32, out_channels=16, )
        self.layer9_2 = _Conv_bn_relu(in_channels=16, out_channels=16, )
        self.layer9_3 = _Conv_bn_relu(in_channels=16, out_channels=16, )

        self.output = nn.Conv3d(in_channels=16, out_channels=num_classes, kernel_size=1)

    def forward(self, x, spacings):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer1 += layer0

        down1 = self.down1(layer1)
        layer2 = self.layer2_1(down1)
        layer2 = self.layer2_2(layer2)
        layer2 += down1

        down2 = self.down2(layer2)
        layer3 = self.layer3_1(down2)
        layer3 = self.layer3_2(layer3)
        layer3 = self.layer3_3(layer3)
        layer3 += down2

        down3 = self.down3(layer3)
        layer4 = self.layer4_1(down3)
        layer4 = self.layer4_2(layer4)
        layer4 = self.layer4_3(layer4)
        layer4 += down3

        # down4 = self.down4(layer4)
        # layer5 = self.layer5_1(down4)
        # layer5 = self.layer5_2(layer5)
        # layer5 = self.layer5_3(layer5)
        # layer5 += down4
        #
        # deconv1 = self.deconv1(layer5)
        # layer6 = self.layer6_1(torch.cat([deconv1, layer4], dim=1))
        # layer6 = self.layer6_2(layer6)
        # layer6 = self.layer6_3(layer6)
        # layer6 += deconv1

        deconv2 = self.deconv2(layer4)
        layer7 = self.layer7_1(torch.cat([deconv2, layer3], dim=1))
        layer7 = self.layer7_2(layer7)
        layer7 = self.layer7_3(layer7)
        layer7 += deconv2

        deconv3 = self.deconv3(layer7)
        layer8 = self.layer8_1(torch.cat([deconv3, layer2], dim=1))
        layer8 = self.layer8_2(layer8)
        layer8 = self.layer8_3(layer8)
        layer8 += deconv3

        deconv4 = self.deconv4(layer8)
        layer9 = self.layer9_1(torch.cat([deconv4, layer1], dim=1))
        layer9 = self.layer9_2(layer9)
        layer9 = self.layer9_3(layer9)
        layer9 += deconv4

        output = self.output(layer9)
        output = output.permute(0, 2, 3, 4, 1).contiguous()

        return output


class VNetStage2(nn.Module):

    def __init__(self, num_classes=1, spacingblock=SpacingBlock2, drop_rate=0.3, training=True):
        super(VNetStage2, self).__init__()
        self.training = training
        # print(self.training)
        self.p = drop_rate
        self.layer0 = _Conv_bn_relu(in_channels=1, out_channels=32)
        self.layer1 = _Conv_bn_relu(in_channels=32, out_channels=32, )

        self.down1 = _Conv_bn_relu(in_channels=32, out_channels=64, stride=(2,2,2), )
        self.layer2_1 = _Conv_bn_relu(in_channels=64, out_channels=64, )
        self.layer2_2 = _Conv_bn_relu(in_channels=64, out_channels=64, )

        self.down2 = _Conv_bn_relu(in_channels=64, out_channels=128, stride=(2,2,2), )
        self.layer3_1 = _Conv_bn_relu(in_channels=128, out_channels=128, )
        self.layer3_2= _Conv_bn_relu(in_channels=128, out_channels=128, )
        self.layer3_3= _Conv_bn_relu(in_channels=128, out_channels=128, )

        self.down3 = _Conv_bn_relu(in_channels=128, out_channels=128, stride=(2,2,2), )
        self.layer4_1 = _Conv_bn_relu(in_channels=128, out_channels=128, )
        self.layer4_2 = _Conv_bn_relu(in_channels=128, out_channels=128, )
        self.layer4_3 = _Conv_bn_relu(in_channels=128, out_channels=128, )

        self.deconv2 = _Deconv_relu(in_channels=128, out_channels=64, kernel=2, stride=2, padding=0)
        self.layer7_1 = _Conv_bn_relu(in_channels=128+64, out_channels=64, )
        self.layer7_2 = _Conv_bn_relu(in_channels=64, out_channels=64, )
        self.layer7_3 = _Conv_bn_relu(in_channels=64, out_channels=64, )

        self.deconv3 = _Deconv_relu(in_channels=64, out_channels=32, kernel=2, stride=2, padding=0)
        self.layer8_1 = _Conv_bn_relu(in_channels=64+32, out_channels=32, )
        self.layer8_2 = _Conv_bn_relu(in_channels=32, out_channels=32, )
        self.layer8_3 = _Conv_bn_relu(in_channels=32, out_channels=32, )

        self.deconv4 = _Deconv_relu(in_channels=32, out_channels=16, kernel=2, stride=2, padding=0)
        self.layer9_1 = _Conv_bn_relu(in_channels=32+16, out_channels=16, )
        self.layer9_2 = _Conv_bn_relu(in_channels=16, out_channels=16, )
        self.layer9_3 = _Conv_bn_relu(in_channels=16, out_channels=16, )

        self.output = nn.Conv3d(in_channels=16, out_channels=num_classes, kernel_size=1)

    def forward(self, x, spacings):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer1 += layer0

        down1 = self.down1(layer1)
        layer2 = self.layer2_1(down1)
        layer2 = self.layer2_2(layer2)
        layer2 += down1

        down2 = self.down2(layer2)
        layer3 = self.layer3_1(down2)
        layer3 = self.layer3_2(layer3)
        layer3 = self.layer3_3(layer3)
        layer3 += down2

        down3 = self.down3(layer3)
        layer4 = self.layer4_1(down3)
        layer4 = self.layer4_2(layer4)
        layer4 = self.layer4_3(layer4)
        layer4 += down3

        # down4 = self.down4(layer4)
        # layer5 = self.layer5_1(down4)
        # layer5 = self.layer5_2(layer5)
        # layer5 = self.layer5_3(layer5)
        # layer5 += down4
        #
        # deconv1 = self.deconv1(layer5)
        # layer6 = self.layer6_1(torch.cat([deconv1, layer4], dim=1))
        # layer6 = self.layer6_2(layer6)
        # layer6 = self.layer6_3(layer6)
        # layer6 += deconv1

        deconv2 = self.deconv2(layer4)
        layer7 = self.layer7_1(torch.cat([deconv2, layer3], dim=1))
        layer7 = self.layer7_2(layer7)
        layer7 = self.layer7_3(layer7)
        layer7 += deconv2

        deconv3 = self.deconv3(layer7)
        layer8 = self.layer8_1(torch.cat([deconv3, layer2], dim=1))
        layer8 = self.layer8_2(layer8)
        layer8 = self.layer8_3(layer8)
        layer8 += deconv3

        deconv4 = self.deconv4(layer8)
        layer9 = self.layer9_1(torch.cat([deconv4, layer1], dim=1))
        layer9 = self.layer9_2(layer9)
        layer9 = self.layer9_3(layer9)
        layer9 += deconv4

        output = self.output(layer9)
        output = output.permute(0, 2, 3, 4, 1).contiguous()

        return output


if __name__ == '__main__':
    import time
    import numpy as np

    t = time.time()
    net = VNetStage2(num_classes=1)
    # net.apply(init)
    x = torch.ones((2, 1, 32, 32, 16))
    spacing = torch.from_numpy(np.asarray([[0.8, 0.8, 0.8], [2.6, 1.2, 2.5]]))
    output = net(x, spacing)
    print(output[0].size())
    print(time.time() - t)

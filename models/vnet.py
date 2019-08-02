"""

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DialResUNet(nn.Module):
    """

    """
    def __init__(self, num_classes=1, training=True):
        super().__init__()

        self.training = training
        self.num_classes = num_classes

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.PReLU(16),

            nn.Conv3d(16, 16, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.PReLU(64),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 1), stride=1, padding=(2, 2, 0), dilation=(2, 2, 1)),
            nn.PReLU(64),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 1), stride=1, padding=(4, 4, 0), dilation=(4, 4, 1)),
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, (3,3,1), 1, padding=(3,3,0), dilation=(3,3,1)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (3,3,1), 1, padding=(4,4,0), dilation=(4,4,1)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (3,3,1), 1, padding=(5,5,0), dilation=(5,5,1)),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, (3,3,1), 1, padding=(1,1,0)),
            nn.PReLU(256),

            nn.Conv3d(256, 256, (3,3,1), 1, padding=(1,1,0)),
            nn.PReLU(256),

            nn.Conv3d(256, 256, (3,3,1), 1, padding=(1,1,0)),
            nn.GroupNorm(num_groups=64, num_channels=256),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, (3,3,1), 1, padding=(1,1,0)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (3,3,1), 1, padding=(1,1,0)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (3,3,1), 1, padding=(1,1,0)),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, (3,3,1), 1, padding=(1,1,0)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (3,3,1), 1, padding=(1,1,0)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (3,3,1), 1, padding=(1,1,0)),
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, (3,3,1), 1, padding=(1,1,0)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (3,3,1), 1, padding=(1,1,0)),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, (2,2,1), (2,2,1), padding=0),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, (2,2,1), (2,2,1), padding=0),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, (2,2,1), (2,2,1), padding=0),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, (3,3,1), (1,1,1), padding=(1,1,0)),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, (2,2,1), (2,2,1), padding=0),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, (2,2,1), (2,2,1), padding=0),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, (2,2,1), (2,2,1), padding=0),
            nn.PReLU(32)
        )

        self.map4 = nn.Conv3d(32, self.num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs, spacings):

        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, 0.3, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, 0.3, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, 0.3, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, 0.3, self.training)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, 0.3, self.training)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.3, self.training)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)
        output4 = output4.permute(0, 2, 3, 4, 1).contiguous()

        return output4


class VNet(nn.Module):
    """
    with out dilation
    """
    def __init__(self, num_classes=1, training=True):
        super().__init__()

        self.training = training
        self.num_classes = num_classes

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.ReLU(16),

            nn.Conv3d(16, 16, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.ReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.ReLU(32),

            nn.Conv3d(32, 32, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.ReLU(32),

        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.ReLU(64),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.ReLU(64),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.ReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, (3,3,1), 1, padding=(1, 1, 0)),
            nn.ReLU(128),

            nn.Conv3d(128, 128, (3,3,1), 1, padding=(1, 1, 0)),
            nn.ReLU(128),

            nn.Conv3d(128, 128, (3,3,1), 1, padding=(1, 1, 0)),
            nn.ReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, (3,3,1), 1, padding=(1,1,0)),
            nn.ReLU(256),

            nn.Conv3d(256, 256, (3,3,1), 1, padding=(1,1,0)),
            nn.ReLU(256),

            nn.Conv3d(256, 256, (3,3,1), 1, padding=(1,1,0)),
            nn.ReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, (3,3,1), 1, padding=(1,1,0)),
            nn.ReLU(128),

            nn.Conv3d(128, 128, (3,3,1), 1, padding=(1,1,0)),
            nn.ReLU(128),

            nn.Conv3d(128, 128, (3,3,1), 1, padding=(1,1,0)),
            nn.ReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, (3,3,1), 1, padding=(1,1,0)),
            nn.ReLU(64),

            nn.Conv3d(64, 64, (3,3,1), 1, padding=(1,1,0)),
            nn.ReLU(64),

            nn.Conv3d(64, 64, (3,3,1), 1, padding=(1,1,0)),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, (3,3,1), 1, padding=(1,1,0)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (3,3,1), 1, padding=(1,1,0)),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, (2,2,1), (2,2,1), padding=0),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, (2,2,1), (2,2,1), padding=0),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, (2,2,1), (2,2,1), padding=0),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, (3,3,1), (1,1,1), padding=(1,1,0)),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, (2,2,1), (2,2,1), padding=0),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, (2,2,1), (2,2,1), padding=0),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, (2,2,1), (2,2,1), padding=0),
            nn.PReLU(32)
        )

        self.map4 = nn.Conv3d(32, self.num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs, spacings):

        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, 0.3, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, 0.3, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, 0.3, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, 0.3, self.training)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, 0.3, self.training)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.3, self.training)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)
        output4 = output4.permute(0, 2, 3, 4, 1).contiguous()

        return output4


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal(module.weight.data, 0.25)
        nn.init.constant(module.bias.data, 0)


if __name__ == '__main__':
    import time
    import numpy as np

    t = time.time()
    net = DialResUNet(training=True, num_classes=1)
    x = torch.ones((2, 1, 128, 128, 16))
    spacing = torch.from_numpy(np.asarray([[0.8, 0.8, 0.8], [2.6, 1.2, 2.5]]))
    output = net(x, spacing)
    print(output[0].size())
    print(time.time() - t)

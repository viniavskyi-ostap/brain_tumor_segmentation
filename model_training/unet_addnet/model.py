import torch.nn as nn
from model_training.common.modules import ResBlock


class UNetAddNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetAddNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, 32, (3, 3), padding=1)
        self.block2 = ResBlock(32)
        self.downsample3 = nn.Conv2d(32, 64, (3, 3), stride=2, padding=1)

        self.block4 = ResBlock(64)
        self.block5 = ResBlock(64)
        self.downsample6 = nn.Conv2d(64, 128, (3, 3), stride=2, padding=1)

        self.block7 = ResBlock(128)
        self.block8 = ResBlock(128)
        self.downsample9 = nn.Conv2d(128, 256, (3, 3), stride=2, padding=1)

        self.block10 = ResBlock(256)
        self.block11 = ResBlock(256)
        self.block12 = ResBlock(256)
        self.block13 = ResBlock(256)

        self.upsample14 = nn.ConvTranspose2d(256, 128, (2, 2), stride=2)
        self.block15 = ResBlock(128)

        self.upsample16 = nn.ConvTranspose2d(128, 64, (2, 2), stride=2)
        self.block17 = ResBlock(64)

        self.upsample18 = nn.ConvTranspose2d(64, 32, (2, 2), stride=2)
        self.block19 = ResBlock(32)

        self.conv20 = nn.Conv2d(32, out_channels, (1, 1))

    def forward(self, X):
        # encoder
        enc32 = self.conv1(X)
        enc32 = self.block2(enc32)
        enc64 = self.downsample3(enc32)

        enc64 = self.block4(enc64)
        enc64 = self.block5(enc64)
        enc128 = self.downsample6(enc64)

        enc128 = self.block7(enc128)
        enc128 = self.block8(enc128)
        enc256 = self.downsample9(enc128)

        enc256 = self.block10(enc256)
        enc256 = self.block11(enc256)
        enc256 = self.block12(enc256)
        enc256 = self.block13(enc256)

        # decoder
        out = self.upsample14(enc256)
        out += enc128
        out = self.block15(out)

        out = self.upsample16(out)
        out += enc64
        out = self.block17(out)

        out = self.upsample18(out)
        out += enc32
        out = self.block19(out)

        out = self.conv20(out)

        return out, enc256


if __name__ == '__main__':
    from torchsummary import summary

    net = UNetAddNet(in_channels=4, out_channels=3)
    summary(net, (4, 240, 240), device='cpu')

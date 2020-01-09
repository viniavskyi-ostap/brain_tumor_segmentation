import torch.nn as nn
from collections import OrderedDict


class ResBlock(nn.Module):

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.channels = channels

        self.block = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm2d(channels)),
            ('relu1', nn.ReLU()),
            ('conv1', nn.Conv2d(channels, channels, (3, 3), padding=1)),
            ('bn2', nn.BatchNorm2d(channels)),
            ('relu2', nn.ReLU()),
            ('conv2', nn.Conv2d(channels, channels, (3, 3), padding=1))
        ]))

    def forward(self, X):
        out = self.block(X)
        return out + X


if __name__ == '__main__':
    from torchsummary import summary

    res = ResBlock(channels=32)

    summary(res, (32, 160, 192), device='cpu')

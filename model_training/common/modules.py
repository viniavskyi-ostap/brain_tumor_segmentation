import torch.nn as nn
import numpy as np
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


class Reshape(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(Reshape, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        if not np.prod(in_shape) == np.prod(out_shape):
            raise ValueError('Elements number mismatch.')

    def forward(self, X):
        return X.view(X.size(0), *self.out_shape)


if __name__ == '__main__':
    from torchsummary import summary

    res = ResBlock(channels=32)

    summary(res, (32, 160, 192), device='cpu')

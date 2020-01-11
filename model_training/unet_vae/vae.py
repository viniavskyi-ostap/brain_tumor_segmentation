import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from model_training.common.modules import Reshape, ResBlock


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.embed = nn.Sequential(OrderedDict([
            ('bn', nn.BatchNorm2d(256)),
            ('relu', nn.ReLU()),
            ('conv', nn.Conv2d(256, 16, (3, 3), stride=2)),
            ('flatten', Reshape((16, 14, 14), (16*14*14, ))),
            ('fc', nn.Linear(16*14*14, 256))
        ]))

        self.dec1 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(128, 16*15*15)),
            ('unflatten', Reshape(16*15*15, (16, 15, 15))),
            ('relu', nn.ReLU()),
            ('conv', nn.Conv2d(16, 256, (1, 1))),
            ('upsample', nn.ConvTranspose2d(256, 256, (2, 2), stride=2))
        ]))

        self.up2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(256, 256, (1, 1))),
            ('upsample', nn.ConvTranspose2d(256, 128, (2, 2), stride=2))
        ]))

        self.block3 = ResBlock(128)

        self.up4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(128, 128, (1, 1))),
            ('upsample', nn.ConvTranspose2d(128, 64, (2, 2), stride=2))
        ]))

        self.block5 = ResBlock(64)

        self.up6 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(64, 64, (1, 1))),
            ('upsample', nn.ConvTranspose2d(64, 32, (2, 2), stride=2))
        ]))

        self.block7 = ResBlock(32)

        self.conv8 = nn.Conv2d(32, 4, (1, 1))

    def forward(self, encoded):
        encoded = self.embed(encoded)

        batch_size = encoded.size(0)
        latent_dim = encoded.size(1) // 2

        sample = torch.randn(batch_size, latent_dim).to(encoded.device)
        mu, sigma = encoded[:, :latent_dim], encoded[:, latent_dim:]
        sample = sample * sigma + mu

        decoded = self.dec1(sample)
        decoded = self.up2(decoded)
        decoded = self.block3(decoded)
        decoded = self.up4(decoded)
        decoded = self.block5(decoded)
        decoded = self.up6(decoded)
        decoded = self.block7(decoded)
        decoded = self.conv8(decoded)

        return decoded, mu, sigma


if __name__ == '__main__':
    from torchsummary import summary

    vae = VAE()
    summary(vae, (256, 30, 30), device='cpu')

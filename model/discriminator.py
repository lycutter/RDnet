import torch
import torch.nn as nn
from spectral import SpectralNorm
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Sequential(SpectralNorm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)),
                                    nn.LeakyReLU(0.2))
        self.conv_2 = nn.Sequential(SpectralNorm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
                                    nn.LeakyReLU(0.2))
        self.conv_3 = nn.Sequential(SpectralNorm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
                                    nn.LeakyReLU(0.2))
        self.conv_4 = nn.Sequential(SpectralNorm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
                                    nn.LeakyReLU(0.2))
        self.conv_5 = nn.Sequential(SpectralNorm(nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)))

    def forward(self, x):

        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        out3 = self.conv_3(out2)
        out4 = self.conv_4(out3)

        out = self.conv_5(out4)


        return out
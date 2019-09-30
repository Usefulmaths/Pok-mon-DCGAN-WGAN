import torch.nn as nn
import math
from layers import DownsampleConv, UpsampleConv


class Generator(nn.Module):
    def __init__(self, image_size, latent_dimension, hidden_channels, output_channels):
        super().__init__()
        log_size = int(math.log2(image_size))

        self.network_modules = nn.ModuleList(
            [UpsampleConv(latent_dimension, 2**log_size * hidden_channels)])

        for i in range(0, log_size - 2):
            self.network_modules.append(UpsampleConv(
                2**(log_size - i) * hidden_channels, 2**(log_size - i - 1) * hidden_channels))
            last_filter = 2**(log_size - i - 1) * hidden_channels

        self.network_modules.append(UpsampleConv(
            last_filter, output_channels, just_conv=True))
        self.network_modules.append(nn.Tanh())

        self.network = nn.Sequential(*self.network_modules)

    def forward(self, z):
        return self.network(z)


class Discriminator(nn.Module):
    def __init__(self, image_size, hidden_channels, input_channels):
        super().__init__()

        log_size = int(math.log2(image_size))

        self.network_modules = nn.ModuleList(
            [DownsampleConv(input_channels, hidden_channels)])

        for i in range(0, log_size - 2):
            self.network_modules.append(DownsampleConv(
                2**i * hidden_channels, 2**(i + 1) * hidden_channels))
            last_filter = 2**(i + 1) * hidden_channels

        self.network_modules.append(
            DownsampleConv(last_filter, 1, just_conv=True))
        self.network_modules.append(nn.Sigmoid())
        self.network = nn.Sequential(*self.network_modules)

    def forward(self, x):
        return self.network(x)


class Critic(nn.Module):
    def __init__(self, image_size, hidden_channels, input_channels):
        super().__init__()

        log_size = int(math.log2(image_size))

        self.network_modules = nn.ModuleList(
            [DownsampleConv(input_channels, hidden_channels)])

        for i in range(0, log_size - 2):
            self.network_modules.append(DownsampleConv(
                2**i * hidden_channels, 2**(i + 1) * hidden_channels))
            last_filter = 2**(i + 1) * hidden_channels

        self.network_modules.append(
            DownsampleConv(last_filter, 1, just_conv=True))
        self.network = nn.Sequential(*self.network_modules)

    def forward(self, x):
        return self.network(x)

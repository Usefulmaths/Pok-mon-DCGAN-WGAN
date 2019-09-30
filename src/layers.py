import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class UpsampleConv(nn.Module):
    def __init__(self, input_filter, output_filter, just_conv=False):
        super().__init__()

        self.input_filter = input_filter
        self.output_filter = output_filter

        self.conv_trans = nn.ConvTranspose2d(
            input_filter, output_filter, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(output_filter)
        self.relu = nn.LeakyReLU(0.2)

        if just_conv:
            self.network_block = nn.Sequential(
                self.conv_trans
            )

        else:
            self.network_block = nn.Sequential(
                self.conv_trans,
                self.batch_norm,
                self.relu
            )

    def forward(self, x):
        # Double image height and width
        x = self.network_block(x)

        return x


class DownsampleConv(nn.Module):
    def __init__(self, input_filter, output_filter, just_conv=False):
        super().__init__()

        self.input_filter = input_filter
        self.output_filter = output_filter

        self.conv = nn.Conv2d(input_filter, output_filter,
                              kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(output_filter)
        self.layer_norm = LayerNorm(output_filter)
        self.relu = nn.LeakyReLU(0.2)

        if just_conv:
            self.network_block = nn.Sequential(
                self.conv
            )

        else:
            self.network_block = nn.Sequential(
                self.conv,
                #        self.batch_norm,
                self.layer_norm,
                self.relu
            )

    def forward(self, x):
        # Double image height and width
        x = self.network_block(x)

        return x

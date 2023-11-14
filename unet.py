import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from itertools import pairwise


class MultiConv(nn.Module):
    """(convolution => [BN] => ReLU) * n"""

    def __init__(self, *channels,
                 kernel_sizes=3, padding=1, stride=1, bias=False, **conv2d_kwargs):
        super().__init__()
        self.n = len(channels) - 1
        if self.n < 1:
            raise ValueException

        if isinstance(kernel_sizes, int):
            kernel_sizes = list(repeat(kernel_sizes, len(channels)-1))
        components = []
        for (input_channels, output_channels), kernel_size in zip(pairwise(channels), kernel_sizes):
            components.extend([
                  nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, bias=bias, **conv2d_kwargs),
                  nn.BatchNorm2d(output_channels),
                  nn.ReLU(inplace=True)
              ])
        self.multiconv = nn.Sequential(*components)

    def forward(self, x):
        return self.multiconv(x)


class Encode(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            MultiConv(in_channels, out_channels, out_channels)
        )
        # print(f'Encode: {(in_channels, out_channels)}')

    def forward(self, x):
        return self.maxpool_conv(x)


class Decode(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # print(f'Decode: {(in_channels, out_channels)}')

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = MultiConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = MultiConv(in_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, depth=4, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth
        self.bilinear = bilinear
        shapes = list(self.shapes)

        self.in_c = MultiConv(n_channels, 64)
        self.encodes = self.create_encodes(shapes)
        self.decodes = self.create_decodes(shapes)
        self.out_c = nn.Conv2d(64, n_classes, kernel_size=1)

    @property
    def factor(self):
        return 2 if self.bilinear else 1

    @property
    def shapes(self):
        return pairwise(map(lambda d: 64*2**d, range(self.depth+1)))

    def create_encodes(self, shapes: list[tuple[int, int]]):
        encodes = []
        for lesser, greater in shapes[:-1]:
            encodes.append(Encode(lesser, greater))
        last_lesser, last_greater = shapes[-1]
        encodes.append(Encode(last_lesser, last_greater // self.factor))
        return nn.ModuleList(encodes)

    def create_decodes(self, shapes: list[tuple[int, int]]):
        decodes = []
        first_lesser, first_greater = shapes[0]
        decodes.append(Decode(first_greater, first_lesser, self.bilinear))
        for lesser, greater in shapes[1:]:
            decodes.append(Decode(greater, lesser // self.factor, self.bilinear))
        return nn.ModuleList(decodes)

    def forward(self, x):
        encoded_xs = [self.in_c(x)]
        for i, encode in enumerate(self.encodes):
          encoded_xs.append(encode(encoded_xs[-1]))
        curr_x = encoded_xs[-1]
        for i in reversed(range(self.depth)):
            curr_x = self.decodes[i](curr_x, encoded_xs[i])
        logits = self.out_c(curr_x)
        return logits

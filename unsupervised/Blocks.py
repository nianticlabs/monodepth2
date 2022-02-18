from torch import nn
from typing import Union, Tuple


class ResConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int, num_blocks: int = 2):
        super(ResConv, self).__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks

        modules = []
        for i in range(num_blocks):
            modules.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size))
            modules.append(nn.ELU())

        self.modules = nn.ModuleList(modules)

    def forward(self, x):
        output = x
        for module in self.convs:
            output = module(output)

        # skip connection
        return x + output


class ConvElu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]]):
        super(ConvElu, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride)
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(self.conv(x))


class UpConvElu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 input_padding: Union[int, Tuple[int, int]],
                 output_padding: Union[int, Tuple[int, int]],
                 activation: bool = True):
        super(UpConvElu, self).__init__()

        self.activation = activation
        self.upconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=input_padding, output_padding=output_padding)
        if self.activation:
            self.elu = nn.ELU()

    def forward(self, x):
        if not self.activation:
            return self.upconv(x)
        return self.elu(self.upconv(x))


class Reshaper(nn.Module):
    def __init__(self, out_shape: Tuple[int, int, int]):
        super(Reshaper, self).__init__()

        # out_shape is a tuple of (C, H, W)
        self.out_shape = out_shape

    def forward(self, x):
        return x.reshape((-1, self.out_shape[0], self.out_shape[1], self.out_shape[2]))

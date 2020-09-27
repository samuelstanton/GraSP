import torch
import torch.nn as nn


class ZeroOp(nn.Module):
    def __init__(self, strides):
        super().__init__()
        assert isinstance(strides, int)
        self.stride = strides

    def forward(self, inputs):
        inputs = inputs[:, :, ::self.stride, ::self.stride]
        return inputs * torch.zeros_like(inputs)


class Identity(nn.Module):
    @staticmethod
    def forward(inputs):
        return inputs


class FactorizedReduce(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 2 == 0
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1,
                                stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1,
                                stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels, track_running_stats=True)

    def forward(self, inputs):
        x = self.relu(inputs)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class ReLUConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation,
                 bias, affine):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels, track_running_stats=True, affine=affine)

    def forward(self, inputs):
        return self.bn(self.conv(self.relu(inputs)))


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, relu, affine):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels, track_running_stats=True, affine=affine),
        ]
        if relu:
            modules.append(nn.ReLU())
        super(ConvBNReLU, self).__init__(*modules)
        self.in_dim = in_channels
        self.out_dim = out_channels
        self.num_conv = 1


class Downsampler(nn.Sequential):
    def __init__(self, in_channels):
        out_channels = 2 * in_channels
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels, track_running_stats=True)
        ]
        super().__init__(*modules)
        self.in_dim = in_channels
        self.out_dim = out_channels


class Flattener(nn.Module):
    def __init__(self):
        super(Flattener, self).__init__()

    def forward(self, inputs):
        return torch.flatten(inputs, start_dim=1)

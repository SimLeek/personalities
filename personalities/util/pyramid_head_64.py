import torch
from torch import nn as nn


class PyramidHead64(nn.Module):
    """A simple 64x64 pyramid image encoder."""

    def __init__(self, layer_channels=(3, 64, 64, 64, 64, 64, 64, 64)):
        """
        A simple 64x64 pyramid image encoder.

        :param layer_channels: determines the channels per layer.
                               First is input layer. Usually 1, 3, or 4.
                               Last is output vector size. Should be large to represent a lot.
                               Middle layers can be hard to figure out.
                               see: https://arxiv.org/pdf/1909.01861.pdf
        """
        super(PyramidHead64, self).__init__()
        assert len(layer_channels) == 6

        self.conv0 = nn.Conv2d(layer_channels[0], layer_channels[1], 4)
        self.conv1 = nn.Conv2d(layer_channels[1], layer_channels[2], 4)
        self.conv2 = nn.Conv2d(layer_channels[2], layer_channels[3], 4)
        self.conv3 = nn.Conv2d(layer_channels[3], layer_channels[4], 4)
        self.conv4 = nn.Conv2d(layer_channels[4], layer_channels[5], 4)
        self.conv5 = nn.Conv2d(layer_channels[5], layer_channels[6], 4)
        self.conv6 = nn.Conv2d(layer_channels[6], layer_channels[7], 4)

        self.pool = nn.AvgPool2d(2)

        self.output_vector_len = \
            layer_channels[0] + layer_channels[1] + layer_channels[2] + \
            layer_channels[3] + layer_channels[6] + layer_channels[7]

    def forward(self, x):
        """
        Run the pyramid head.

        :param x: a tensor representing a 64x64 image.
        :returns a vector with length equal to the number of channels in the
                 input, 1st, 2nd, 3rd, 6th, and 7th layers combined.
                 default output size is 323
        """
        a_pyramid = list()
        a_pyramid.append(x[:, :, 12:52, 12:52])
        a_pyramid.append(self.pool(x))
        for _ in range(4):
            a_pyramid.append(self.pool(a_pyramid[-1]))
        a_pyramid.append(self.pool(x))
        b_pyramid = [self.conv0(a_pyramid[x]) for x in range(len(a_pyramid) - 1)]
        b_pyramid[-2] = nn.functional.pad(b_pyramid[-2], [1, 1, 1, 1], mode='constant', value=0)
        c_pyramid = [self.conv1(b_pyramid[x]) for x in range(len(b_pyramid) - 1)]
        d_pyramid = [self.conv2(c_pyramid[x]) for x in range(len(c_pyramid) - 1)]
        e_pyramid = [self.conv3(d_pyramid[x]) for x in range(len(d_pyramid) - 1)]
        f_pyramid = [self.conv4(x) for x in e_pyramid]
        f_pyramid[-1] = nn.functional.pad(f_pyramid[-1], [1, 1, 1, 1], mode='constant', value=0)
        g_pyramid = [self.conv5(x) for x in f_pyramid]
        h_pyramid = [self.conv6(g_pyramid[0])]

        last_xs = [
            a_pyramid[-1],
            b_pyramid[-1],
            c_pyramid[-1],
            d_pyramid[-1],
            g_pyramid[-1],
            h_pyramid[-1]
        ]

        out_x = torch.cat(last_xs, -1)

        return out_x

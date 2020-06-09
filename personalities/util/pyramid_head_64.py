import kornia
import torch
from torch import nn as nn


def image_to_pyramid(input_image: torch.Tensor, batch_size=1, min_size=(4, 4), scaling_factor=2):
    width = input_image.shape[-1]
    height = input_image.shape[-2]

    center: torch.Tensor = torch.tensor(
        input_image[width, height], dtype=torch.float32).view(1, 2) / 2. - 0.5
    center = center.expand(batch_size, -1)

    sx = sy = torch.tensor([0] * batch_size)

    angle = torch.zeros(batch_size, 1)

    scale = torch.ones(batch_size) * (1.0 / scaling_factor)

    params = dict(translations=torch.zeros(batch_size, 2),
                  center=center,
                  scale=scale,
                  angle=angle,
                  sx=sx,
                  sy=sy,
                  resample=torch.tensor(0),
                  align_corners=torch.tensor(False))

    pyramid = [input_image]
    while any(m > i for m, i in zip(min_size, pyramid[-1].shape)):
        pyramid.append(
            kornia.augmentation.F.apply_affine(pyramid[-1], params)
        )
    return pyramid


class PyramidHead64(nn.Module):
    """A simple 64x64 pyramid image encoder."""

    def __init__(self, layer_channels=(3, 64, 64, 64, 64), scaling_factor=2):
        """
        A simple 64x64 pyramid image encoder.

        :param layer_channels: determines the channels per layer.
                               First is input layer. Usually 1, 3, or 4.
                               Last is output vector size. Should be large to represent a lot.
                               Middle layers can be hard to figure out.
                               see: https://arxiv.org/pdf/1909.01861.pdf
        :param num_pyramids: number of pyramids to convolve while keeping same size.
        """
        super(PyramidHead64, self).__init__()

        self.pyramid_pool = nn.AvgPool2d(2)
        self.final_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.pyramid_convs = nn.ModuleList()
        for i in range(len(layer_channels) - 1):
            self.pyramids.append(nn.Conv2d(layer_channels[i], layer_channels[i + 1], 3))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Run the pyramid head.

        :param x: a tensor representing an image. Not an image pyramid. Will be converted to image pyramid on GPU.
        :returns a vector with layer_channels[-1]*num_pyramids neurons
        """
        # todo: for the final pyramids, everything not going into the output can be removed.
        #  So, the last pyramid can have two images, one 1x1 and one 3x3. The next can have three, and so on.
        #  This could lead to significant speedup in some cases.
        x_pyramid = image_to_pyramid(x, scaling_factor=self.scaling_factor)

        xs = [x_pyramid]
        xs[-1][-1] = self.final_pool(xs[-1][-1])
        for e, c in enumerate(self.pyramid_convs):
            xs.append([])
            for i, p in enumerate(xs[-2]):
                if e > 0 and i > 0:
                    last = p
                    if i == len(xs[-2]) - 1:
                        pooled_x = self.final_pool(xs[-1][i - 1])
                    else:
                        pooled_x = self.pyramid_pool(xs[-1][i - 1])
                    current_x = torch.cat((last, pooled_x), -1)
                else:
                    current_x = xs[e - 1][i]
                convd_x = c(current_x)

                xs[-1].append(convd_x)

        last_xs = []
        for x in xs:
            last_xs.append(x[-1])

        out_x = torch.cat(last_xs, -1)

        return out_x

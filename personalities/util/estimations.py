import math as m
from torch.nn.modules.utils import _pair


def ensure_2d(*inps):
    outs = []
    for inp in inps:
        inp = _pair(inp)
        outs.append(inp)
    return tuple(outs)


def conv2d_output_shape(input_shape, out_channels: int, kernel_size, stride=1, padding=0, dilation=1):
    # codified from: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
    padding, dilation, kernel_size, stride = ensure_2d(padding, dilation, kernel_size, stride)

    h_out = m.floor((input_shape[2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w_out = m.floor((input_shape[3] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    output_shape = (input_shape[0], out_channels, h_out, w_out)
    return output_shape


def conv_transpose_2d_output_shape(input_shape, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                                   dilation=1):
    # codified from: https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d
    kernel_size, stride, padding, output_padding, dilation = ensure_2d(kernel_size, stride, padding, output_padding,
                                                                       dilation)

    h_out = (input_shape[2] - 1) * stride[0] - 2 * padding[0] + \
            dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1

    w_out = (input_shape[3] - 1) * stride[1] - 2 * padding[1] + \
            dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1

    output_shape = (input_shape[0], out_channels, h_out, w_out)
    return output_shape

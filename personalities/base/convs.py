import torch.nn as nn
import torch
import math
from torch import functional as F
from torch._six import container_abcs
from itertools import repeat


class Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(self, x, weight, weight_fluidity):
        '''
            The forward computation for a convolution function

            Arguments:
            X -- output activations of the previous layer, numpy array of shape (n_H_prev, n_W_prev) assuming input channels = 1
            W -- Weights, numpy array of size (f, f) assuming number of filters = 1

            Returns:
            H -- conv output, numpy array of size (n_H, n_W)
            cache -- cache of values needed for conv_backward() function
            '''
        # https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199

        # Retrieving dimensions from X's shape
        (_, n_H_prev, n_W_prev, _) = x.shape

        # Retrieving dimensions from W's shape
        (_, f, f, _) = weight.shape

        # Compute the output dimensions assuming no padding and stride = 1
        n_H = n_H_prev - f + 1
        n_W = n_W_prev - f + 1

        # Initialize the output H with zeros
        H = torch.zeros((n_H, n_W))

        # Looping over vertical(h) and horizontal(w) axis of output volume
        for h in range(n_H):
            for w in range(n_W):
                x_slice = x[h:h + f, w:w + f]
                H[h, w] = torch.sum(x_slice * weight)

        # Saving information in 'cache' for backprop
        self.save_for_backward(x, weight, weight_fluidity)

        return H

    @staticmethod
    def backward(self, dy):
        # https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
        '''
            The backward computation for a convolution function

            Arguments:
            dH -- gradient of the cost with respect to output of the conv layer (H), numpy array of shape (n_H, n_W) assuming channels = 1
            cache -- cache of values needed for the conv_backward(), output of conv_forward()

            Returns:
            dX -- gradient of the cost with respect to input of the conv layer (X), numpy array of shape (n_H_prev, n_W_prev) assuming channels = 1
            dW -- gradient of the cost with respect to the weights of the conv layer (W), numpy array of shape (f,f) assuming single filter
            '''

        # Retrieving information from the "cache"
        (X, W, ws) = self.saved_tensors

        # Retrieving dimensions from X's shape
        (n_H_prev, n_W_prev) = X.shape

        # Retrieving dimensions from W's shape
        (f, f) = W.shape

        # Retrieving dimensions from dH's shape
        (n_H, n_W) = dy.shape

        # Initializing dX, dW with the correct shapes
        dX = torch.zeros(X.shape)
        dW = torch.zeros(W.shape)

        # Looping over vertical(h) and horizontal(w) axis of the output
        for h in range(n_H):
            for w in range(n_W):
                dX[h:h + f, w:w + f] += W * dy(h, w)
                dW += X[h:h + f, w:w + f] * dy(h, w)

        return dX, dW

x = torch.tensor([[[[3.], [4.]], [[5.], [6.]]]], requires_grad=True)
w = torch.tensor([[[[1.], [2.]], [[1.], [2.]]]], requires_grad=True)
wf = torch.tensor([[[[.5], [.1]], [[1], [.2]]]], requires_grad=True)
m = Conv2d.apply(x, w, wf)
print(m)
m.backward()
print(x.grad.data)
print(w.grad.data)
print(wf.grad.data)


'''
class _ConvNd(nn.Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            weight_solidifying_rate=.001,
            sparse_activation_percentage=.02
    ):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight_solidifying_rate = weight_solidifying_rate
        self.sparse_activation_percentage = sparse_activation_percentage
        if transposed:
            tensor_init = in_channels, out_channels // groups, *kernel_size
        else:
            tensor_init = out_channels, in_channels // groups, *kernel_size
        self.weight = nn.Parameter(
            torch.Tensor(*tensor_init)
        )
        self.weight_importance = nn.parameter(
            torch.Tensor(*tensor_init)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.weight_importance)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size:
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    .. include:: cudnn_deterministic.rst

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)
'''

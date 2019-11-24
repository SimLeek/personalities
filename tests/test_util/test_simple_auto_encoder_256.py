import mock
from personalities.util import simple_auto_encoder_256 as sae256
from personalities.util.estimations import conv2d_output_shape, conv_transpose_2d_output_shape


def test_autoencoder_256_init():
    ae = sae256.AutoEncoder(movement_data_len=1024)

    a = conv2d_output_shape((1, 3, 256, 256), ae.conv1.out_channels, ae.conv1.kernel_size, padding=ae.conv1.padding,
                            stride=ae.conv1.stride)
    b = conv2d_output_shape(a, ae.conv2.out_channels, ae.conv2.kernel_size, padding=ae.conv2.padding,
                            stride=ae.conv2.stride)
    c = conv2d_output_shape(b, ae.conv3.out_channels, ae.conv3.kernel_size, padding=ae.conv3.padding,
                            stride=ae.conv3.stride)
    d = conv2d_output_shape(c, ae.conv4.out_channels, ae.conv4.kernel_size, padding=ae.conv4.padding,
                            stride=ae.conv4.stride)

    e = conv_transpose_2d_output_shape(d, ae.unconv1.out_channels, ae.unconv1.kernel_size)
    f = conv_transpose_2d_output_shape(e, ae.unconv2.out_channels, ae.unconv2.kernel_size, padding=ae.unconv2.padding,
                                       stride=ae.unconv2.stride)
    g = conv_transpose_2d_output_shape(f, ae.unconv3.out_channels, ae.unconv3.kernel_size, padding=ae.unconv3.padding,
                                       stride=ae.unconv3.stride)
    h = conv_transpose_2d_output_shape(g, ae.unconv4.out_channels, ae.unconv4.kernel_size, padding=ae.unconv4.padding,
                                       stride=ae.unconv4.stride)

    assert a == (1, 20, 52, 52)
    assert b == (1, 50, 11, 11)
    assert c == (1, 100, 3, 3)
    assert d == (1, 200, 1, 1)
    assert e == (1, 612, 3, 3)
    assert f == (1, 306, 11, 11)
    assert g == (1, 148, 53, 53)
    assert h == (1, 3, 256, 256)

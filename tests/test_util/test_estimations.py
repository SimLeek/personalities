from personalities.util.estimations import ensure_2d, conv2d_output_shape, conv_transpose_2d_output_shape


def test_ensure_2d():
    a = 1
    b = 2
    c = (3, 3)
    a, b, c = ensure_2d(a, b, c)
    assert a == (1, 1)
    assert b == (2, 2)
    assert c == (3, 3)


def test_conv2d_output_shape():
    a = conv2d_output_shape((1, 3, 600, 800), 20, 5, padding=2, stride=5)
    b = conv2d_output_shape(a, 50, 5, padding=2, stride=5)
    c = conv2d_output_shape(b, 100, 5, padding=2, stride=5)
    d = conv2d_output_shape(c, 200, (5, 7))

    assert a == (1, 20, 120, 160)
    assert b == (1, 50, 24, 32)
    assert c == (1, 100, 5, 7)
    assert d == (1, 200, 1, 1)


def test_ConvTranspose2d_output_shape():
    a = conv_transpose_2d_output_shape((1, 200, 1, 1), 100, 3)
    b = conv_transpose_2d_output_shape(a, 50, 5, padding=2, stride=5)
    c = conv_transpose_2d_output_shape(b, 20, 5, padding=1, stride=5)
    d = conv_transpose_2d_output_shape(c, 3, 4, padding=4, stride=5)

    assert a == (1, 100, 3, 3)
    assert b == (1, 50, 11, 11)
    assert c == (1, 20, 53, 53)
    assert d == (1, 3, 256, 256)

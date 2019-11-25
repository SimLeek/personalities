from personalities.util.cv_helpers import *


def test_cv_image_to_pytorch():
    # color
    image_rgb = np.zeros((8, 8, 3))

    pytorch_rgb = cv_image_to_pytorch(image_rgb)

    assert pytorch_rgb.shape == torch.Size([1, 3, 8, 8])
    assert pytorch_rgb.is_cuda

    pytorch_rgb = cv_image_to_pytorch(image_rgb, cuda=False)

    assert pytorch_rgb.shape == torch.Size([1, 3, 8, 8])
    assert not pytorch_rgb.is_cuda

    # black and white
    image_bw = np.zeros((8, 8))

    pytorch_bw = cv_image_to_pytorch(image_bw)

    assert pytorch_bw.shape == torch.Size([1, 1, 8, 8])
    assert pytorch_bw.is_cuda

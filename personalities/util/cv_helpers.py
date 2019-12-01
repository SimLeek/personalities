import numpy as np
import torch


# todo: move this to displayarray or relayarray
def cv_image_to_pytorch(image: np.ndarray, cuda: bool = True) -> torch.Tensor:
    torch_image = torch.from_numpy(image).float()
    if cuda:
        torch_image = torch_image.cuda()
    if torch_image.ndim == 2:
        torch_image = torch_image[..., None]
    torch_image = torch_image.permute((2, 0, 1))
    torch_image = torch_image[None, ...]
    return torch_image


def pytorch_image_to_cv(arr: torch.Tensor) -> np.ndarray:
    np_img = arr.detach().cpu()
    np_img = np_img[0, ...].squeeze()
    np_img = np_img.permute((1, 2, 0))
    return np_img.numpy()


# todo: move this to displayarray or relayarray
def vector_to_2d_encoding(vector: np.ndarray, cuda: bool = True) -> torch.Tensor:
    torch_vec = torch.from_numpy(vector).float()
    if cuda:
        torch_vec = torch_vec.cuda()
    torch_vec = torch_vec[None, :, None, None]
    return torch_vec

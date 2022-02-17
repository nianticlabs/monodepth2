import torch
import torch.nn.functional as F
from typing import Tuple

SOBEL_Y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).double().reshape((1, 1, 3, 3))
SOBEL_X = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).double().reshape((1, 1, 3, 3))

# we won't ever change these so we don't need to compute gradients for them
SOBEL_X.requires_grad = False
SOBEL_Y.requires_grad = False


def compute_appx_spatial_gradients(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes approximate spatial gradients using sobel filters.

    :param input: The tensor for which to approximate gradients
    :return: a tuple of tensors with approximate x gradient values and approximate y gradient values
    """

    if len(input.shape) == 3:
        # add channel dimension
        input = input.unsqueeze(dim=1)

    batch_size, channels, height, width = input.shape

    sobel_x = SOBEL_X.expand(channels, 1, 3, 3)
    sobel_y = SOBEL_Y.expand(channels, 1, 3, 3)

    grad_x = F.conv2d(input, SOBEL_X, padding="valid", groups=channels)
    grad_y = F.conv2d(input, SOBEL_Y, padding="valid", groups=channels)

    return grad_x, grad_y

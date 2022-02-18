from typing import Optional

from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch

from unsupervised.SpatialGradientApproximation import compute_appx_spatial_gradients


PURPLE = np.array([0.5, 0, 0.5])
BLUE = np.array([0, 0.5, 0.5])

"""
This file's only functional purpose is to give the reader a bit of a playground to work with
some of the interesting components of the unsupervised depth loss, like F.grid_sample()
"""

def make_circle(height: int, width: int, radius: int) -> np.ndarray:
    arr = np.zeros((height, width, 3))
    arr[:, :, :] = PURPLE.reshape((1, 1, 3))

    center_x = width/4
    center_y = height/4

    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            dist_sq_from_center = (x - center_x) ** 2 + (y - center_y) ** 2
            if radius * radius >= dist_sq_from_center:
                arr[y, x, :] = BLUE

    return arr


def make_disparity(height: int, width: int, x_shift: int) -> np.ndarray:
    disparity = np.zeros((height, width, 2))

    shift_frac = 2*x_shift/width

    disparity[:, :, 1] = np.linspace(-1, 1, height).reshape((height, 1))
    disparity[:, :, 0] = np.linspace(-1, 1, width).reshape((1, width))

    # apply shift, remember this is altering the corresponding location in the original image, so if this index
    # is larger, it means that the larger index in the original image will occur earlier in the sample
    disparity[:, :, 0] -= shift_frac

    # (H, W, 2)
    return disparity


def apply_disparity_w_grid_sample(input: np.ndarray, grid: np.ndarray):
    input = torch.Tensor(input)
    grid = torch.Tensor(grid)

    # permute input so it's ordered correctly (H, W, C) -> (C, H, W)
    input = input.permute((2, 0, 1))#torch.permute(input, (2, 0, 1))

    # add batch dim to both
    input = input.unsqueeze(dim=0)
    grid = grid.unsqueeze(dim=0)

    # apply grid sampling
    new_image = F.grid_sample(input, grid)

    # remove batch dim
    new_image = new_image.squeeze()

    # permute input so it's ordered correctly (C, H, W) -> (H, W, C)
    new_image = new_image.permute((1, 2, 0))# torch.permute(new_image, (1, 2, 0))

    return new_image.cpu().detach().numpy()


def display_arr(arr: np.ndarray):
    plt.imshow(arr)
    plt.show()


def visualize_differentiable_indexing():
    circle = make_circle(1000, 1000, 100)
    display_arr(circle)

    grid = make_disparity(1000, 1000, 500)
    shifted_circle = apply_disparity_w_grid_sample(circle, grid)
    display_arr(shifted_circle)


def display_tensor(x: torch.Tensor, title: Optional[str] = None):
    x = x.squeeze().permute((1, 2, 0))

    x = x.cpu().detach().numpy()

    plt.figure()
    plt.imshow(x)
    if title is not None:
        plt.title(title)
    plt.show()


def visualize_spatial_gradients():
    circle = make_circle(1000, 1000, 100)
    circle[800:900, :, :] = BLUE.reshape((1, 1, 3))
    display_arr(circle)

    # re-format 'circle' into correct dims
    circle = torch.tensor(circle).permute((2, 0, 1)).float().unsqueeze(dim=0)

    grad_x, grad_y = compute_appx_spatial_gradients(circle)

    display_tensor(grad_x, "grad_x")
    display_tensor(grad_y, "grad_y")


if __name__ == "__main__":
    visualize_differentiable_indexing()

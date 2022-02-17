import torch.nn.functional as F
import torch
from typing import Tuple

# the library we use for SSIM is called 'kornia' and it requires a fairly recent torch version,
# hopefully that doesn't break the dataloader components in this repo.
# you can install it with 'pip install kornia'
from kornia.losses.ssim import ssim_loss

from unsupervised.SpatialGradientApproximation import compute_appx_spatial_gradients


def reconstruct_input_from_disp_maps(
        stereo_pair: Tuple[torch.Tensor, torch.Tensor],
        disp_maps: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstructs input from disparity maps
    :param stereo_pair: a tuple of the original left and right images as tensors
    :param disp_maps: a tuple of disparity maps, left-to-right then right-to-left
    :return: a tuple of reconstructions, left image then right image
    """

    batch_size, height, width = disp_maps[0].shape

    # disp maps only provide horizontal disparities since images are rectified, but
    # F.grid_sample expects x and y disparities, so we need to add an axis to the disp maps
    left_to_right_disp = torch.zeros((batch_size, height, width, 2))
    right_to_left_disp = torch.zeros_like(left_to_right_disp)  # left and right disp maps should have the same shape

    # copy disparities into x position
    left_to_right_disp[:, :, :, 0] = disp_maps[0]
    right_to_left_disp[:, :, :, 0] = disp_maps[1]

    # set y position disparities to values that do not cause any transformation of the input image
    # if the use of 'torch.linspace()' is confusing, check out the documentation for torch.functional.grid_sample()
    left_to_right_disp[:, :, :, 1] = torch.linspace(-1, 1, height).reshape((1, height, 1))
    right_to_left_disp[:, :, :, 1] = torch.linspace(-1, 1, height).reshape((1, height, 1))

    left_image, right_image = stereo_pair

    # use F.grid_sample() to reconstruct left and right images from disparities
    left_reconstruction = F.grid_sample(right_image, right_to_left_disp)
    right_reconstruction = F.grid_sample(left_image, left_to_right_disp)

    return left_reconstruction, right_reconstruction


def loss(
        stereo_pair: Tuple[torch.Tensor, torch.Tensor],
        predicted_disparities: Tuple[torch.Tensor, torch.Tensor],
        reconstructions: Tuple[torch.Tensor, torch.Tensor],
        reconstruction_weight: float = 1.0,
        alpha_reconstruction: float = 0.85,
        disp_smoothness_weight: float = 0.1,
        lr_disp_consistency_weight: float = 1.0):
    """
    Loss function from https://arxiv.org/pdf/1609.03677.pdf
    :param stereo_pair: a tuple of the original left and right images as tensors
    :param predicted_disparities: a tuple of the predicted left-to-right and right-to-left disparities
    :param reconstructions: a tuple of reconstructions, left image then right image
    :return: loss to be optimized with gradient descent
    """

    # TODO: if adding more disparity map resolutions, must edit this to compute loss at all scales

    left_image, right_image = stereo_pair
    left_to_right_disp, right_to_left_disp = predicted_disparities
    left_recon, right_recon = reconstructions

    # reconstruction loss term (called 'appearance matching loss' in paper)
    reconstruction_term = reconstruction_loss(left_image, left_recon, alpha_reconstruction) +\
        reconstruction_loss(right_image, right_recon, alpha_reconstruction)

    loss = reconstruction_weight * reconstruction_term

    # disparity smoothness loss term
    disp_smoothness_term = disp_smoothness_loss(left_image, left_to_right_disp) +\
        disp_smoothness_loss(right_image, right_to_left_disp)
    loss += disp_smoothness_weight * disp_smoothness_term

    # left-right disparity consistency loss
    lr_disp_consistency_term = lr_disp_consistency_loss(left_to_right_disp, right_to_left_disp) +\
        lr_disp_consistency_loss(right_to_left_disp, left_to_right_disp)
    loss += lr_disp_consistency_weight * lr_disp_consistency_term

    # TODO: experiment with adding a supervised loss term as well
    return loss


def reconstruction_loss(img1: torch.Tensor, img2: torch.Tensor, alpha: float):
    ssim = ssim_loss(img1, img2, max_val=1.0, window_size=3, reduction="none", padding="valid")

    return alpha*0.5*torch.sum(torch.mean((1-ssim), dim=0)) +\
                (1 - alpha)*torch.sum(torch.mean(torch.abs(img1 - img2), dim=0))


def disp_smoothness_loss(img: torch.Tensor,
                         disp: torch.Tensor):
    # paper doesn't really define how to implement spatial gradients of grids,
    # so I used sobel filters to approximate
    img_gradx, img_grady = compute_appx_spatial_gradients(img)
    disp_gradx, disp_grady = compute_appx_spatial_gradients(disp)

    # take L1 norm of img_gradx and img_grady, like in paper
    img_gradx_l1 = torch.sum(torch.abs(img_gradx), dim=1)
    img_grady_l1 = torch.sum(torch.abs(img_grady), dim=1)

    loss = torch.abs(disp_gradx)*torch.exp(-1*img_gradx_l1) + torch.abs(disp_grady)*torch.exp(-1*img_grady_l1)
    loss = torch.sum(torch.mean(loss, dim=0))
    return loss


def lr_disp_consistency_loss(disp1: torch.Tensor, disp2: torch.Tensor):
    """
    Takes opposite disparity maps as input
    """
    # the following pseudocode uses the more traditional convention for
    # disparity maps (disparity of an image to itself is zero), but
    # the parameters we are provided are not like that

    # loss = 0
    #
    # for x in range(3):
    #     for y in range(4):
    #         loss += abs(disp1[y, x] - disp2[y, x + disp1[x, y]])
    #
    # loss /= N

    batch_size, height, width = disp1.shape

    disp1 = disp1.unsqueeze(dim=1)
    disp2 = disp2.unsqueeze(dim=1)

    # need to provide y disparity for grid_sample
    grid = torch.zeros((batch_size, height, width, 2))
    grid[:, :, :, 0] = disp1
    grid[:, :, :, 1] = torch.linspace(-1, 1, height).reshape((1, height, 1))

    # we have to use grid_sample because indexing like this is not differentiable
    projected_disp2 = F.grid_sample(disp2, grid)

    loss = torch.abs(disp1 - projected_disp2)
    loss = torch.sum(torch.mean(loss, dim=0))
    return loss

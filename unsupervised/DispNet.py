import torch
from torch import nn

from typing import Tuple


class DispNet(nn.Module):
    # described in https://arxiv.org/pdf/1512.02134.pdf
    # used in https://arxiv.org/pdf/1609.03677.pdf

    def __init__(self, dropout_p: float):
        super(DispNet, self).__init__()

        self.dropout_p = dropout_p

        # TODO: more

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: in the unsupervised monocular depth paper, they output disp at 4 different scales. Should we do this?

        :param x: an input tensor of shape (N, C, H, W) where N is batch size,
            C is # channels (e.g., RGB = 3), H and W are height and width
            IMPORTANT: RGB input is expected to be floating point between 0 and 1
        :return: a tuple of two disparity maps, left-to-right disp
            and right-to-left disp (dimensions determined by KITTI).
            These maps will have elements in [-1, 1] and can be transformed
            into a more conventional view of disparity by subtracting
            torch.linspace(-1, 1, width) from each column and then
            multiplying by width to get a number of pixels
        """

        # TODO: I need to know the input dimensions and the depth dimensions before implementing this
        raise NotImplementedError("")

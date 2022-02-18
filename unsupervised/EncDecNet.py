import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
from collections import deque

from unsupervised.Blocks import ConvElu, UpConvElu, Reshaper

MAX_DISP_FRAC = 0.3  # from the paper


class EncDecNet(nn.Module):
    # take the place of the bowtie-looking thing in the diagrams of https://arxiv.org/pdf/1609.03677.pdf

    def __init__(self):
        super(EncDecNet, self).__init__()

        internal_blocks = []
        skip_down_blocks = []
        skip_up_blocks = []

        # start with 3 x 375 x 1242
        # first, get down to a reasonable resolution

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

        # use a deque to implement skip connections
        skip_tensor_deque = deque()

        for module in self.skip_down_blocks:
            x = module(x)
            skip_tensor_deque.append(x)

        for module in self.internal_blocks:
            x = module(x)

        for module in self.skip_up_blocks:
            skip = skip_tensor_deque.pop()
            x = module(x + skip)

        # TODO: can modify this to obtain multiple scales of depth
        return MAX_DISP_FRAC * F.sigmoid(x)

    def _init_model(self):
        self.skip_down_blocks = nn.ModuleList([
            ConvElu(3, 4, 5, 2),
            ConvElu(4, 4, 5, 2),
            ConvElu(4, 8, 5, 2),

            # after this point, we wouldn't extract depth maps
            ConvElu(8, 16, (3, 7), (2, 3)),
            ConvElu(16, 32, (3, 7), (2, 2)),
            ConvElu(32, 4, (3, 7), (1, 1))
        ])
        self.skip_up_blocks = nn.ModuleList([
            UpConvElu(4, 32, (3, 7), 1, 0, 0),
            UpConvElu(32, 16, (3, 7), 1, 0, 0),
            UpConvElu(16, 8, (3, 7), (2, 3), 0, 1),
            UpConvElu(8, 4, 5, 2, 0, (0, 1)),
            UpConvElu(4, 4, 5, 2, 0, (1, 0)),
            UpConvElu(4, 1, 5, 2, 0, (0, 1), activation=False)
        ])
        self.internal_blocks = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            Reshaper((1, 8, 16)),
            UpConvElu(1, 4, 1, 1, 0, 0)
        ])




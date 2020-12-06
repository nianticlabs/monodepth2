from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset

class NYUDataset(MonoDataset):
    """NYU Depth v2 Dataset"""
    def __init__(self, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)

        # Camera intrinsics matrix, normalized by original image size
        # First row divided by width, second row divided by height
        self.K = np.array([[518.8579/640, 0, 325.5834/640, 0],
                           [0, 519.4696/480, 253.7361/480, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (640, 480)

    def check_depth(self):
        """Returns if depth maps exist or not"""
        return True

    def get_color(self, folder, frame_index, side, do_flip):
        """Gets the color image corresponding to the frame index in the
        specified folder.
        """
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        """Returns the path string of the image corresponding to the frame_index
        in the specified folder.
        """
        image_path = os.path.join(self.data_path, folder, str(frame_index) + self.img_ext)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        """Gets the depth map corresponding to the frame index in the specified
        folder.
        """
        image_path = os.path.join(self.data_path, folder, str(frame_index)+".png")
        depth_gt = pil.open(image_path)
        depth_gt = np.asarray(depth_gt) / 255 * 10

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

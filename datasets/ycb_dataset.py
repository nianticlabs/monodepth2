import os

import numpy as np
import PIL.Image as pil
import scipy.io as sio

from .mono_dataset import MonoDataset


class YCBDataset(MonoDataset):

    # def __init__(self, data_root_path, train_file_path, transform=None):
    def __init__(self, *args, **kwargs):
        super(YCBDataset, self).__init__(*args, **kwargs)

    def get_color(self, folder, frame_index, side, do_flip):
        img_path = os.path.join(self.data_path, folder, "{0:06d}-color.png".format(frame_index))
        color = self.loader(img_path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        self.set_K(folder, frame_index)

        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        depth_path = os.path.join(self.data_path, folder, "{0:06d}-depth.png".format(frame_index))

        meta_data = self.get_metadata(folder, frame_index)
        factor_depth = meta_data["factor_depth"]

        depth = pil.open(depth_path)

        if do_flip:
            depth = depth.transpose(pil.FLIP_LEFT_RIGHT)

        depth = np.array(depth) / factor_depth

        return depth

    def check_depth(self):
        line = self.filenames[0].split()
        folder = line[0]
        frame_index = int(line[1])

        depth_path = os.path.join(self.data_path, folder, "{0:06d}-depth.png".format(frame_index))

        return os.path.isfile(depth_path)

    def get_metadata(self, folder, frame_index):
        meta_path = os.path.join(self.data_path, folder, "{0:06d}-meta.mat".format(frame_index))
        meta_data = sio.loadmat(meta_path)
        return meta_data

    def set_K(self, folder, frame_index):
        """
        Since the intrinsics can vary per image, we set the correct intrinsics per image directly.
        """
        meta_data = self.get_metadata(folder, frame_index)
        intrinsics = meta_data['intrinsic_matrix']
        # K is expected to be a 4x4 (homogeneous) matrix
        K = np.eye(4)
        K[0:3, 0:3] = intrinsics

        self.K = K.astype(np.float32)

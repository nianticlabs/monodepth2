# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import random
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms

cv2.setNumThreads(0)


from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
import numpy as np


class DGPDataset(data.Dataset):
    """Superclass for monocular dataloaders
    """
    RAW_HEIGHT = 1216
    RAW_WIDTH = 1936
    def __init__(self,
                 data_path,
                 split,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 datum_names,
                 back_context,
                 forward_context
                 ):
        super(DGPDataset, self).__init__()

        self.data_path = data_path
        self.split = split
        self.height = height
        self.width = width
        self.num_scales = num_scales

        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = True if self.split=='train' else False

        self.datum_names=datum_names
        self.back_context=back_context
        self.forward_context=forward_context

        self.dataset=SynchronizedSceneDataset(self.data_path,
                                              split=self.split,
                                              datum_names=self.datum_names,
                                              backward_context=self.back_context,
                                              forward_context=self.forward_context,
                                              generate_depth_from_datum= None if self.split=='train' else 'lidar'
        )

        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.05
            # self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth=False


    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)

                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
    def __len__(self):
        return len(self.dataset)

    def load_intrinsics(self, index):
        intrinsics = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]).astype(np.float32)
        intrinsics[:3,:3] = self.dataset[index][0][0]['intrinsics']
        intrinsics[0, :] /= self.RAW_WIDTH
        intrinsics[1, :] /= self.RAW_HEIGHT
        return intrinsics

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth_gt"                              for ground truth depth maps

        <frame_id> is:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        poses = {}

        for i in self.frame_idxs:
            inputs[("color", i, -1)]=self.dataset[index][i+1][0]['rgb']

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.load_intrinsics(index)

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            # color_aug = transforms.ColorJitter.get_params(
            #     self.brightness, self.contrast, self.saturation, self.hue)

            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth and False:
            depth_gt = self.dataset[index][0][0]['depth']
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs


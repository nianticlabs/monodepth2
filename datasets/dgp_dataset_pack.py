# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
import numpy as np

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.utils.camera import Camera, generate_depth_map
from dgp.utils.pose import Pose

from packnet_sfm.utils.misc import make_list
from packnet_sfm.utils.types import is_tensor, is_numpy, is_list

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def stack_sample(sample):
    """Stack a sample from multiple sensors"""
    # If there is only one sensor don't do anything
    if len(sample) == 1:
        return sample[0]

    # Otherwise, stack sample
    stacked_sample = {}
    for key in sample[0]:
        # Global keys (do not stack)
        if key in ['idx', 'dataset_idx', 'sensor_name', 'filename']:
            stacked_sample[key] = sample[0][key]
        else:
            # Stack torch tensors
            if is_tensor(sample[0][key]):
                stacked_sample[key] = torch.stack([s[key] for s in sample], 0)
            # Stack numpy arrays
            elif is_numpy(sample[0][key]):
                stacked_sample[key] = np.stack([s[key] for s in sample], 0)
            # Stack list
            elif is_list(sample[0][key]):
                stacked_sample[key] = []
                # Stack list of torch tensors
                if is_tensor(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            torch.stack([s[key][i] for s in sample], 0))
                # Stack list of numpy arrays
                if is_numpy(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            np.stack([s[key][i] for s in sample], 0))

    # Return stacked sample
    return stacked_sample

########################################################################################################################
#### DATASET
########################################################################################################################

class DGPDataset:
    """
    DGP dataset class

    Parameters
    ----------
    path : str
        Path to the dataset
    split : str {'train', 'val', 'test'}
        Which dataset split to use
    cameras : list of str
        Which cameras to get information from
    depth_type : str
        Which lidar will be used to generate ground-truth information
    with_pose : bool
        If enabled pose estimates are also returned
    with_semantic : bool
        If enabled semantic estimates are also returned
    back_context : int
        Size of the backward context
    forward_context : int
        Size of the forward context
    data_transform : Function
        Transformations applied to the sample
    """
    def __init__(self, path, split,
                 cameras=None,
                 depth_type=None,
                 input_depth_type=None,
                 with_pose=False,
                 with_semantic=False,
                 back_context=0,
                 forward_context=0,
                 data_transform=None,
                 ):
        self.path = path
        self.split = split
        self.dataset_idx = 0

        self.bwd = back_context
        self.fwd = forward_context
        self.has_context = back_context + forward_context > 0

        self.num_cameras = len(cameras)
        self.data_transform = data_transform

        self.depth_type = depth_type
        self.with_depth = depth_type is not None and depth_type is not ''
        self.with_pose = with_pose
        self.with_semantic = with_semantic

        self.input_depth_type = input_depth_type
        self.with_input_depth = input_depth_type is not None and input_depth_type is not ''

        self.dataset = SynchronizedSceneDataset(path,
            split=split,
            datum_names=cameras,
            backward_context=back_context,
            forward_context=forward_context,
            requested_annotations=None,
            only_annotated_datums=False,
        )

    def generate_depth_map(self, sample_idx, datum_idx, filename):
        """
        Generates the depth map for a camera by projecting LiDAR information.
        It also caches the depth map following DGP folder structure, so it's not recalculated

        Parameters
        ----------
        sample_idx : int
            sample index
        datum_idx : int
            Datum index
        filename :
            Filename used for loading / saving

        Returns
        -------
        depth : np.array [H, W]
            Depth map for that datum in that sample
        """
        # Generate depth filename
        filename = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('depth/{}'.format(self.depth_type)))
        # Load and return if exists
        if os.path.exists(filename):
            return np.load(filename)['depth']
        # Otherwise, create, save and return
        else:
            # Get pointcloud
            scene_idx, sample_idx_in_scene, _ = self.dataset.dataset_item_index[sample_idx]
            pc_datum_idx_in_sample = self.dataset.get_datum_index_for_datum_name(
                scene_idx, sample_idx_in_scene, self.depth_type)
            pc_datum_data = self.dataset.get_point_cloud_from_datum(
                scene_idx, sample_idx_in_scene, pc_datum_idx_in_sample)
            # Create camera
            camera_rgb = self.get_current('rgb', datum_idx)
            camera_pose = self.get_current('pose', datum_idx)
            camera_intrinsics = self.get_current('intrinsics', datum_idx)
            camera = Camera(K=camera_intrinsics, p_cw=camera_pose.inverse())
            # Generate depth map
            world_points = pc_datum_data['pose'] * pc_datum_data['point_cloud']
            depth = generate_depth_map(camera, world_points, camera_rgb.size[::-1])
            # Save depth map
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, depth=depth)
            # Return depth map
            return depth

    def get_current(self, key, sensor_idx):
        """Return current timestep of a key from a sensor"""
        return self.sample_dgp[self.bwd][sensor_idx][key]

    def get_backward(self, key, sensor_idx):
        """Return backward timesteps of a key from a sensor"""
        return [] if self.bwd == 0 else \
            [self.sample_dgp[i][sensor_idx][key] \
             for i in range(0, self.bwd)]

    def get_forward(self, key, sensor_idx):
        """Return forward timestep of a key from a sensor"""
        return [] if self.fwd == 0 else \
            [self.sample_dgp[i][sensor_idx][key] \
             for i in range(self.bwd + 1, self.bwd + self.fwd + 1)]

    def get_context(self, key, sensor_idx):
        """Get both backward and forward contexts"""
        return self.get_backward(key, sensor_idx) + self.get_forward(key, sensor_idx)

    def get_filename(self, sample_idx, datum_idx):
        """
        Returns the filename for an index, following DGP structure

        Parameters
        ----------
        sample_idx : int
            Sample index
        datum_idx : int
            Datum index

        Returns
        -------
        filename : str
            Filename for the datum in that sample
        """
        scene_idx, sample_idx_in_scene, datum_indices = self.dataset.dataset_item_index[sample_idx]
        scene_dir = self.dataset.dataset_metadata.directory
        filename = self.dataset.get_datum(
            scene_idx, sample_idx_in_scene, datum_indices[datum_idx]).datum.image.filename
        return os.path.splitext(os.path.join(os.path.basename(scene_dir),
                                             filename.replace('rgb', '{}')))[0]

    def __len__(self):
        """Length of dataset"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a dataset sample"""
        # Get DGP sample (if single sensor, make it a list)
        self.sample_dgp = self.dataset[idx]
        self.sample_dgp = [make_list(sample) for sample in self.sample_dgp]

        # Loop over all cameras
        sample = []
        for i in range(self.num_cameras):
            data = {
                'idx': idx,
                'dataset_idx': self.dataset_idx,
                'sensor_name': self.get_current('datum_name', i),
                #
                'filename': self.get_filename(idx, i),
                'splitname': '%s_%010d' % (self.split, idx),
                #
                'rgb': self.get_current('rgb', i),
                'intrinsics': self.get_current('intrinsics', i),
            }

            # If depth is returned
            if self.with_depth:
                data.update({
                    'depth': self.generate_depth_map(idx, i, data['filename'])
                })

            # If depth is returned
            if self.with_input_depth:
                data.update({
                    'input_depth': self.generate_depth_map(idx, i, data['filename'])
                })

            # If pose is returned
            if self.with_pose:
                data.update({
                    'extrinsics': self.get_current('extrinsics', i).matrix,
                    'pose': self.get_current('pose', i).matrix,
                })

            # If context is returned
            if self.has_context:
                data.update({
                    'rgb_context': self.get_context('rgb', i),
                })
                # If context pose is returned
                if self.with_pose:
                    # Get original values to calculate relative motion
                    orig_extrinsics = Pose.from_matrix(data['extrinsics'])
                    orig_pose = Pose.from_matrix(data['pose'])
                    data.update({
                        'extrinsics_context':
                            [(orig_extrinsics.inverse() * extrinsics).matrix
                             for extrinsics in self.get_context('extrinsics', i)],
                        'pose_context':
                            [(orig_pose.inverse() * pose).matrix
                             for pose in self.get_context('pose', i)],
                    })

            sample.append(data)

        # Apply same data transformations for all sensors
        if self.data_transform:
            sample = [self.data_transform(smp) for smp in sample]

        # Return sample (stacked if necessary)
        return stack_sample(sample)

########################################################################################################################


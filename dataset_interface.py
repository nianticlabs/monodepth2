import torch
from torchvision import transforms
from typing import Tuple
from PIL import Image
import os
import numpy as np
from kitti_utils import generate_depth_map

def to_depth(disparity : torch.Tensor, baseline : torch.Tensor, focalLength : torch.Tensor) -> torch.Tensor:
    depth = (baseline * focalLength)/disparity
    return depth

def to_disparity(depth : torch.Tensor, baseline : torch.Tensor, focalLength : torch.Tensor) -> torch.Tensor:
    disparity = (baseline * focalLength)/depth
    return disparity

def read_calib_file(path):
    float_chars = set("0123456789.e+- ")

    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value.split()
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array([float(num) for num in value.split(' ')])
                except ValueError:
                    pass  # casting error: data[key] already eq. value, so pass
    return data

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, type : str):
        #path to drive for data
        basedir = 'kitti_data'
        date = '2011_09_26'
        self.calibDir = os.path.join(basedir, date)
        driveFiles = [f.path  for f in os.scandir(self.calibDir) if f.is_dir()]
        numDrives = len(driveFiles)
        #percentage splits of train, test, eval
        splits = [9/10, 1/20, 1/20]
        assert sum(splits) == 1
        #physical numbers
        numTrain : int  = int(splits[0]*numDrives)
        numTest : int   = int(splits[1]*numDrives)
        numEval : int   = numDrives - numTrain - numTest
        if type == "train":
            driveDirs = driveFiles[0:numTrain]
        elif type == "test":
            driveDirs = driveFiles[numTrain:numTrain+numTest]
        elif type == "eval":
            driveDirs = driveFiles[numTrain+numTest:]
        
        #paths to data
        cam2DirPaths = [os.path.join(os.path.join(drivepath, 'image_02'), 'data') for drivepath in driveDirs]
        cam3DirPaths = [os.path.join(os.path.join(drivepath, 'image_03'), 'data') for drivepath in driveDirs]
        veloDirPaths = [os.path.join(os.path.join(drivepath, 'velodyne_points'), 'data') for drivepath in driveDirs]
        
        print(f"cam2dirpaths len {len(cam2DirPaths)} cam 3 paths len {len(cam3DirPaths)} velo paths len {len(veloDirPaths)} ")

        #file name lists
        self.cam2Files = []
        self.cam3Files = []
        self.veloFiles = []
        for index in range(0, len(cam2DirPaths)):
            curr2Dir = [file.path for file in os.scandir(cam2DirPaths[index])]
            curr3Dir = [file.path for file in os.scandir(cam3DirPaths[index])]
            currVDir = [file.path for file in os.scandir(veloDirPaths[index])]
            if len(curr2Dir) == (len(currVDir)) and len(curr3Dir) == (len(currVDir)):
                self.cam2Files += curr2Dir
                self.cam3Files += curr3Dir
                self.veloFiles += currVDir
        print(f"loaded {len(self.cam2Files)}, {len(self.cam3Files)}, {len(self.veloFiles)} images for {type}")
        #retrive calibration data
        cam2cam = read_calib_file(os.path.join(self.calibDir, "calib_cam_to_cam.txt"))
        P_rectL = cam2cam['P_rect_02'].reshape(3, 4)
        P_rectR = cam2cam['P_rect_03'].reshape(3, 4)
        self.L_Kmat = torch.Tensor(cam2cam['K_02'].reshape(3,3))
        self.R_Kmat = torch.Tensor(cam2cam['K_03'].reshape(3,3))
        self.focalLength = torch.Tensor(self.L_Kmat[0, 0])


        # Compute the rectified extrinsics from cam0 to camN
        T2 = np.eye(4)
        T2[0, 3] = P_rectL[0, 3] / P_rectL[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rectR[0, 3] / P_rectR[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        velo2cam = read_calib_file(os.path.join(self.calibDir, 'calib_velo_to_cam.txt'))
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        T_cam0_velo = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
        T_cam2_velo = T2.dot(T_cam0_velo)
        T_cam3_velo = T3.dot(T_cam0_velo)

        p_cam = np.array([0, 0, 0, 1])
        p_velo2 = np.linalg.inv(T_cam2_velo).dot(p_cam)
        p_velo3 = np.linalg.inv(T_cam3_velo).dot(p_cam)
        self.baseline = torch.Tensor([np.linalg.norm(p_velo3 - p_velo2)])   # rgb baseline

        

    def __len__(self):
        return len(self.cam2Files)

    def __getitem__(self, index) -> tuple: #(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        cam 2 = left cam color
        cam 3 = right cam color
        """
        #get images
        imgL : Image = Image.open(self.cam2Files[index])
        imgR : Image = Image.open(self.cam3Files[index])

        #conversion
        convert_tensor = transforms.ToTensor()
        imgL : torch.Tensor = convert_tensor(imgL).float()     #tensor
        imgR : torch.Tensor = convert_tensor(imgR).float()     #tensor

        #retrieve depth data
        depth_gtL = generate_depth_map(self.calibDir, velo_filename=self.veloFiles[index], cam = 2)
        depth_gtR = generate_depth_map(self.calibDir, velo_filename=self.veloFiles[index], cam = 3)

        #convert to tensor
        depth_gtL : torch.Tensor = torch.Tensor(depth_gtL)
        depth_gtR : torch.Tensor = torch.Tensor(depth_gtR)
               
        

        return (imgL, imgR, depth_gtL, depth_gtR)
                
        
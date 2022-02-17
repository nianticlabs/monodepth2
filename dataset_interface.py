import torch
from torchvision import transforms
from typing import Tuple
from PIL import Image
import os
import numpy as np
from kitti_utils import generate_depth_map

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
                    print(list(float(value.split(' '))))
                    data[key] = np.array(map(float, value.split(' ')))
                except ValueError:
                    pass  # casting error: data[key] already eq. value, so pass
    return data

class MyDataset(torch.Dataset):
    def __init__(self, type : str):
        if type == "train":
            drive = "0001"
        elif type == "test":
            drive = "0002"
        elif type == "eval":
            drive = "0003"
        basedir = 'kitti_data'
        date = '2011_09_26'
        #path to drive for data
        self.calibDir = os.path.join(basedir, date)
        drivepath = os.path.join(calibDir, f"{date}_drive_{drive}_sync")
        #paths to data
        cam2DirPath = os.path.join(os.path.join(drivepath, 'image_02'), 'data')
        cam3DirPath = os.path.join(os.path.join(drivepath, 'image_03'), 'data')
        veloDirPath = os.path.join(os.path.join(drivepath, 'velodyne_points'), 'data')
        #file name lists
        self.cam2Files = [os.path.join(cam2DirPath, file) for file in os.listdir(cam2DirPath)]
        self.cam3Files = [os.path.join(cam3DirPath, file) for file in os.listdir(cam3DirPath)]
        self.veloFiles = [os.path.join(veloDirPath, file) for file in os.listdir(veloDirPath)]

        #retrive calibration data
        cam2cam = read_calib_file(os.path.join(self.calibdir, "calib_cam_to_cam.txt"))
        P_rectL = cam2cam['P_rect_02'].reshape(3,4)
        P_rectR = cam2cam['P_rect_03'].reshape(3, 4)
        P0, P1, P2 = P_rectL
        Q0, Q1, Q2 = P_rectR
        
        # create disp transform
        T = np.array([P0, P1, P0 - Q0, P2])
        convert_tensor = transforms.ToTensor()
        self.dispTranformation : torch.Tensor = convert_tensor(T.T)

    def __len__(self):
        return len(self.cam2Files)
    
    def __getitem__(self, index) -> Tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        cam 2 = left cam color
        cam 3 = right cam color
        """
        #get images
        imgL : Image = Image.open(self.cam2Files[index])
        imgR : Image = Image.open(self.cam3Files[index])
        
        #conversion
        convert_tensor = transforms.ToTensor()
        imgL : torch.Tensor = convert_tensor(imgL)     #tensor
        imgR : torch.Tensor = convert_tensor(imgR)     #tensor

        #retrieve depth data
        depth_gtL = generate_depth_map(self.calibdir, velo_filename=self.veloFiles[index], cam = 2)
        depth_gtR = generate_depth_map(self.calibdir, velo_filename=self.veloFiles[index], cam = 3)
        
        #convert to tensor
        depth_gtL : torch.Tensor = convert_tensor(depth_gtL)
        depth_gtR : torch.Tensor = convert_tensor(depth_gtR)

        dispTransformation = self.dispTranformation

        return (imgL, imgR, depth_gtL, depth_gtR, dispTransformation)
        
                
        
import argparse
import os
from torch.utils.data import DataLoader
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import torch
import time


from dataset_interface import MyDataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)

    parser.add_argument('--visualize-dir', type=str,
                        help="location of directory to save visualized results to")

    return parser.parse_args()


def make_dir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.mkdir(dir)

def calculateDisparityTest():
    # Size of images is 3 x 375 x 1242
    imgL = np.random.randint(low=5,high=255,size=(3,375,1242)).astype(np.uint8)
    imgR = np.random.randint(low=5,high=255,size=(3,375,1242)).astype(np.uint8)

    imgLGray = 0.2989 * imgL[0,:,:] + 0.5870 * imgL[1,:,:] + 0.1140 * imgL[2,:,:]
    imgRGray = 0.2989 * imgR[0, :, :] + 0.5870 * imgR[1, :, :] + 0.1140 * imgR[2, :, :]

    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgLGray.astype(np.uint8), imgRGray.astype(np.uint8))

    return disparity

def calculateDisparity(tup):
    imgL, imgR, depth_gtL, depth_gtR = tup
    imgL = imgL.cpu().detach().numpy()
    imgR = imgR.cpu().detach().numpy()
    depth_gtL = depth_gtL.cpu().detach().numpy()
    depth_gtR = depth_gtR.cpu().detach().numpy()

    imgLGray = 0.2989 * imgL[0, :, :] + 0.5870 * imgL[1, :, :] + 0.1140 * imgL[2, :, :]
    imgRGray = 0.2989 * imgR[0, :, :] + 0.5870 * imgR[1, :, :] + 0.1140 * imgR[2, :, :]

    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgLGray.astype(np.uint8), imgRGray.astype(np.uint8))

    return disparity

def convertDisparityArrayToTensor(disparity) -> torch.tensor:
    return torch.tensor(disparity)

def main():
        # Size of images is 375 x 1242
        args = get_args()

        train_loader = DataLoader(dataset=MyDataset("train"), batch_size=args.batch_size)
        val_loader = DataLoader(dataset=MyDataset("eval"), batch_size=args.batch_size)
        test_loader = DataLoader(dataset=MyDataset("test"), batch_size=args.batch_size)

        for tup in train_loader:
            disparity = calculateDisparity(tup)
            plt.imshow(disparity, 'gray')
            plt.show()



def mainTest():
    start_time = time.time()
    disparity = calculateDisparityTest()
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.imshow(disparity, 'gray')
    plt.show()

if __name__ == "__main__":
    mainTest()


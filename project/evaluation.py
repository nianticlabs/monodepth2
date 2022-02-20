import torch
import argparse
import os
from torch.utils.data import DataLoader

from dataset_interface import *

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-save-dir', type=str, required=True)
    parser.add_argument('--tbx-log-dir', type=str, required=True)
    parser.add_argument('--initial-lr', type=float, default=1e-4)
    parser.add_argument('--num-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)

    return parser.parse_args()


def make_dir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.mkdir(dir)


    # Given a disparity map, convert to depth map and then apply one of the below metrics compared to the ground truth depth map
    # Iterate over the entire dataloader output and average the results from the below
    # Only need to convert disparities (which we have from outputs) to depth, not the other way around

def calculate_quantitative_results_RMS(calculated_disparity_map: torch.Tensor, ground_truth_depth: torch.Tensor):
    testDataset = MyDataset()
    calculated_depth = testDataset.to_depth(calculated_disparity_map, testDataset.baseline, testDataset.focalLength)
    calculated_depth = calculated_depth.cpu().detach().numpy()
    ground_truth_depth = ground_truth_depth.cpu().detach().numpy()
    RMSE = np.sqrt(np.mean((calculated_depth - ground_truth_depth) ** 2))
    return RMSE

def get_RMS_total(all_pairs_of_calcualted_disparity_maps_and_ground_truths):
    totalRMSE = 0
    n = 0
    for tup in all_pairs_of_calcualted_disparity_maps_and_ground_truths:
        calculated_disparity_map, ground_truth_depth = tup
        currentRMSE = calculate_quantitative_results_RMS(calculated_disparity_map, ground_truth_depth)
        totalRMSE = totalRMSE + currentRMSE
        n = n + 1
    return totalRMSE / n





def calculate_quantitaive_results_SILog(calculated_disparity_map: torch.Tensor, ground_truth_depth: torch.Tensor):
    testDataset = MyDataset()
    calculated_depth = testDataset.to_depth(calculated_disparity_map, testDataset.baseline, testDataset.focalLength)
    calculated_depth = calculated_depth.cpu().detach().numpy()
    ground_truth_depth = ground_truth_depth.cpu().detach().numpy()
    di = np.log(calculated_depth) - np.log(ground_truth_depth)
    diSquared = di ** 2
    firstTerm = np.mean(diSquared)
    secondTerm = (np.mean(di)) ** 2
    SILog = firstTerm - secondTerm
    return SILog

def get_SILog_total(all_pairs_of_calcualted_disparity_maps_and_ground_truths):
    totalSILog = 0
    n = 0
    for tup in all_pairs_of_calcualted_disparity_maps_and_ground_truths:
        calculated_disparity_map, ground_truth_depth = tup
        currentSILog = calculate_quantitative_results_RMS(calculated_disparity_map, ground_truth_depth)
        totalSILog = totalSILog + currentSILog
        n = n + 1
    return totalSILog / n


# TODO Integrate with various outputs

import dataset_interface
import pdb
import torch


from dataset_interface import MyDataset, to_depth, to_disparity



def main():
    dataset = MyDataset("train")

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    
    for tup in loader:
        imgL, imgR, depth_gtL, depth_gtR, focalLength, baseline = tup
        depth = to_depth(1, baseline, focalLength)
        print(f"depth from disp = 1: {depth}")
        disparity = to_disparity(depth, baseline, focalLength)
        print(f"to disparity using calced (should be 1): {disparity}")       
        pdb.set_trace()


if __name__ == "__main__":
    main()
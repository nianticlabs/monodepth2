import dataset_interface
import pdb
import torch


from dataset_interface import MyDataset, to_depth, to_disparity



def main():
    dataset = MyDataset("train")

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    depth = to_depth(1, dataset.baseline, dataset.focalLength)
    print(f"depth from disp = 1: {depth}")
    disparity = to_disparity(depth, dataset.baseline, dataset.focalLength)
    print(f"to disparity using calced (should be 1): {disparity}")       

    for tup in loader:

        pdb.set_trace()


if __name__ == "__main__":
    main()
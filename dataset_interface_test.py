import dataset_interface
import pdb
import torch


from dataset_interface import MyDataset, to_depth, to_disparity

class batchData:
    imgL : torch.tensor

def custom_collate(imgL, imgR, depth_gtL, depth_gtR, focalLength, baseline):
    return dataset_interface.Data_Tuple(imgL, imgR, depth_gtL, depth_gtR, focalLength, baseline)

def main():
    dataset = MyDataset("train")

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn = custom_collate)
    i = 0
    for tup in loader:
        print(tup.baseline)
        print(tup.imgR)
        i+=1
        print(f"I have completed {i} iterations out of {len(loader)}")
    


if __name__ == "__main__":
    main()
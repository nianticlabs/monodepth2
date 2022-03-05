import dataset_interface
import pdb
import torch


from dataset_interface import MyDataset, to_depth, to_disparity, get_dataloader

class batchData:
    imgL : torch.tensor



def main():
    #dataset = MyDataset("train")
    #dataset = MyDataset("test")
    
    type = "test"
    batch_size = 3
    shuffle = True
    loader = get_dataloader(type, batch_size, shuffle)
    i = 0
    for tup in loader:
        print(tup.focalLength)
        print(tup.imgR)
        i+=1
        print(f"I have completed {i} iterations out of {len(loader)}")
    


if __name__ == "__main__":
    main()
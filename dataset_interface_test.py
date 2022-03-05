import dataset_interface
import pdb
import torch


from dataset_interface import MyDataset, to_depth, to_disparity, custom_collate

class batchData:
    imgL : torch.tensor



def main():
    #dataset = MyDataset("train")
    dataset = MyDataset("test")
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=8, shuffle=False)#, collate_fn = custom_collate)
    i = 0
    for tup in loader:
        print(tup)
        pdb.set_trace()
        i+=1
        print(f"I have completed {i} iterations out of {len(loader)}")
    


if __name__ == "__main__":
    main()